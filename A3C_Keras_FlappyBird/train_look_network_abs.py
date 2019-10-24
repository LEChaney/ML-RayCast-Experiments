import numpy as np
import sys
from datetime import datetime
sys.path.append("game/")

from coord import CoordinateChannel2D

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input, Concatenate
from keras.layers import Conv2D
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

from keras.layers import Lambda
from keras.backend import slice

import pygame
from pygame import Rect, Color
import wrapped_flappy_bird as game

import scipy.misc
import scipy.stats as st

import threading
import time
import math

import matplotlib.pyplot as plt

GAMMA = 0.99                #discount value
BETA = 0.01                 #regularisation coefficient
IMAGE_ROWS = 96
IMAGE_COLS = 96
ZOOM = 2
NUM_CROPS = 3
TIME_SLICES = 4
EXTRA_ACTIONS = 2
NUM_NORMAL_ACTIONS = 1
NUM_ACTIONS = NUM_NORMAL_ACTIONS + EXTRA_ACTIONS
IMAGE_CHANNELS = TIME_SLICES * NUM_CROPS
LEARNING_RATE = 1e-4
LOSS_CLIPPING = 0.2
LOOK_SPEED = 0.1
# TEMPERATURE = 0
# TEMP_INCR = 1e-6

EPOCHS = 3
THREADS = 1
T_MAX = 15
BATCH_SIZE = 80
T = 0
EPISODE = 0

episode_r = np.empty((0, 1), dtype=np.float32)
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_action_state = np.zeros((0, NUM_ACTIONS * TIME_SLICES))
episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
episode_pred = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
episode_critic = np.empty((0, 1), dtype=np.float32)

# now = datetime.now().strftime("%Y%m%d-%H%M%S")
# summary_writer = tf.summary.FileWriter("logs/look_abs/" + now, tf.get_default_graph())

DUMMY_ADVANTAGE = np.zeros((1, 1))
DUMMY_OLD_PRED  = np.zeros((1, NUM_ACTIONS * 2))

ACTIONS = 2
a_t = np.zeros(ACTIONS)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#loss function for policy output
def logloss(advantage):     #policy loss
	def logloss_impl(y_true, y_pred):
		num_actions = y_pred.shape[1] // 2
		mu = y_pred[:,:num_actions]
		sigma_sq = y_pred[:,num_actions:]
		pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(y_true - mu) / (2. * sigma_sq))
		log_pdf = K.log(pdf + K.epsilon())
		entropy = 0.5 * (K.log(2. * np.pi * sigma_sq + K.epsilon()) + 1.)
		losses = -log_pdf * advantage - BETA * entropy
		return K.mean(losses)# + K.sum(K.min(K.abs(y_true), 1) - 1)
	return logloss_impl

def ppo_loss(advantage, old_pred):
	def loss(y_true, y_pred):
		mu_pred = y_pred[:,:NUM_ACTIONS]
		var_pred = y_pred[:,NUM_ACTIONS:]
		mu_old_pred = old_pred[:,:NUM_ACTIONS]
		var_old_pred = old_pred[:,NUM_ACTIONS:]
		denom = K.sqrt(2 * np.pi * var_pred)
		denom_old = K.sqrt(2 * np.pi * var_old_pred)
		prob_num = K.exp(- K.square(y_true - mu_pred) / (2 * var_pred))
		old_prob_num = K.exp(- K.square(y_true - mu_old_pred) / (2 * var_old_pred))

		prob = prob_num/denom
		old_prob = old_prob_num/denom_old
		r = prob/(old_prob + 1e-10)

		surr1 = r * advantage
		surr2 = K.clip(r, (1 - LOSS_CLIPPING), (1 + LOSS_CLIPPING)) * advantage
		aloss = -K.mean(K.minimum(surr1, surr2))

		entropy = 0.5 * (K.log(2. * np.pi * var_pred + K.epsilon()) + 1.)
		entropy_penalty = -BETA * K.mean(entropy)

		# shape = [K.shape(y_pred)[0], NUM_ACTIONS]
		# eps = K.random_normal(shape)
		# actions = mu_pred + K.sqrt(var_pred) * eps
		# energy_penalty = 0.1 * K.mean(K.square(actions))

		return aloss + entropy_penalty# + energy_penalty
	return loss

#loss function for critic output
# def sumofsquares(y_true, y_pred):        #critic loss
# 	return K.mean(K.square(y_pred - y_true))

#function buildmodel() to define the structure of the neural network in use 
def buildmodel():
	print("Model building begins")

	model = Sequential()
	# keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	AS = Input(shape = (NUM_ACTIONS * TIME_SLICES,), name = 'Action_State')
	h0 = CoordinateChannel2D()(S)
	h0 = Conv2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu', bias_initializer = 'random_uniform')(h0)
	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu', bias_initializer = 'random_uniform')(h1)
	h2 = Flatten()(h1)

	a = Dense(128, bias_initializer = 'random_uniform')(h2)
	b = Dense(128, bias_initializer = 'random_uniform')(AS)
	h2 = Concatenate()([a, b])
	
	h3 = Dense(256, activation = 'relu', bias_initializer = 'random_uniform') (h2)
	P_mu = Dense(NUM_ACTIONS, activation = 'tanh', bias_initializer = 'random_uniform') (h3)
	P_sigma = Dense(NUM_ACTIONS, activation = 'softplus', bias_initializer = 'random_uniform') (h3)
	P = Concatenate(name = 'o_P')([P_mu, P_sigma])
	V = Dense(1, name = 'o_V') (h3)
	
	A = Input(shape = (1,), name = 'Advantage')
	O = Input(shape = (NUM_ACTIONS * 2,), name = 'Old_Prediction')
	model = Model(inputs = [S,AS,A,O], outputs = [P,V])
	# optimizer = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	optimizer = Adam(lr = LEARNING_RATE)
	model.compile(loss = {'o_P': ppo_loss(A,O), 'o_V': 'mse'}, loss_weights = {'o_P': 1., 'o_V' : 1}, optimizer = optimizer)
	return model

#function to preprocess an image before giving as input to the neural network
def preprocess(image, look_action=np.array((0, 0))):
	def crop(img, look_action, crop_height, crop_width):
		crop_height = min(int(crop_height), img.shape[0])
		crop_width = min(int(crop_width), img.shape[1])
		min_y = crop_height // 2
		min_x = crop_width // 2
		max_y = max(img.shape[0] - crop_height//2, min_y)
		max_x = max(img.shape[1] - crop_width//2, min_x)
		range_y = max_y - min_y
		range_x = max_x - min_x

		img = skimage.color.rgb2gray(img)
		y_norm, x_norm = (look_action + 1) / 2
		y = min_y + y_norm * range_y
		x = min_x + x_norm * range_x
		y = int(np.clip(y, min_y, max_y))
		x = int(np.clip(x, min_x, max_x))
		return img[y-crop_height//2:y+crop_height//2, x-crop_width//2:x+crop_width//2]
	
	if len(look_action) == 0:
		look_action = np.array((0, 0))

	images = np.empty((1, IMAGE_ROWS, IMAGE_COLS, NUM_CROPS))
	for i in range(0, NUM_CROPS):
		max_dim = max(image.shape[0], image.shape[1])
		img = crop(image, look_action, max_dim // (ZOOM**i), max_dim // (ZOOM**i))
		img = skimage.transform.resize(img, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')
		if img.min() != img.max(): # Prevent NaNs
			img = skimage.exposure.rescale_intensity(img, out_range=(-1,1))
		img = img.reshape(1, img.shape[0], img.shape[1])
		images[:,:,:,i] = img

	return images

# initialize a new model using buildmodel() or use load_model to resume training an already trained model
model = buildmodel()
model.load_weights("saved_models/look_abs/model_updates8220")
model._make_predict_function()
graph = tf.get_default_graph()

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('o_P').output)


a_t[0] = 1 #index 0 = no flap, 1= flap
#output of network represents probability of flap

game_state = []
look_targets = []
max_score = 0
for i in range(0,THREADS):
	game_state.append(game.GameState(30000))
	look_targets.append(np.array((0., 0.)))


def runprocess(thread_id, s_t, action_state):
	global T
	global a_t
	global model
	global LOOK_SPEED
	global max_score

	t = 0
	t_start = t
	terminal = False
	r_t = 0
	r_store = np.empty((0, 1), dtype=np.float32)
	state_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	action_store = np.empty((0, NUM_ACTIONS), dtype=np.float32)
	action_state_store = np.zeros((0, NUM_ACTIONS * TIME_SLICES))
	pred_store = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
	critic_store = np.empty((0, 1), dtype=np.float32)
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
	action_state = action_state.reshape((1, -1))

	while t-t_start < T_MAX and terminal == False:
		t += 1
		T += 1
		# LOOK_SPEED += TEMP_INCR
		# LOOK_SPEED = np.clip(LOOK_SPEED, 0, 0.1)

		with graph.as_default():
			out = model.predict([s_t, action_state, DUMMY_ADVANTAGE, DUMMY_OLD_PRED])[0][0]

		num_actions = out.shape[0] // 2
		mu = out[:num_actions]
		sigma_sq = out[num_actions:]
		eps = np.random.randn(mu.shape[0])
		actions = mu + np.sqrt(sigma_sq) * eps
		look_action = actions[1:]

		# no = np.random.rand()
		# a_t = [0,1] if no < actions[0] else [1,0]  #stochastic action
		a_t = [0,1] if 0.5 < actions[0] else [1,0]  #deterministic action

		x_t, r_t, terminal = game_state[thread_id].frame_step(a_t)
		max_score = max(max_score, game_state[thread_id].score)

		if len(look_action) == 0:
			look_action = np.array((0, 0))
		look_targets[thread_id] = look_action

		# Invalid action penalty
		# valid = (-1 <= actions) & (actions <= 1)
		# valid = np.append(valid, (-1 <= look_targets[thread_id]) & (look_targets[thread_id] <=1))
		# invalid = ~valid
		# r_t += np.sum(invalid * np.abs(np.append(actions, look_targets[thread_id]))) * -0.1

		look_targets[thread_id] = np.clip(look_targets[thread_id], -1., 1.)

		# Visualize crops
		# TODO: Tidy
		for i in range(NUM_CROPS):
			max_dim = max(x_t.shape[0], x_t.shape[1])
			crop_height = min(int(max_dim // (ZOOM**i)), x_t.shape[0])
			crop_width = min(int(max_dim // (ZOOM**i)), x_t.shape[1])
			min_y = crop_height // 2
			min_x = crop_width // 2
			max_y = max(x_t.shape[0] - crop_height//2, min_y)
			max_x = max(x_t.shape[1] - crop_width//2, min_x)
			range_y = max_y - min_y
			range_x = max_x - min_x

			y_norm, x_norm = (look_targets[thread_id] + 1) / 2
			y = min_y + y_norm * range_y
			x = min_x + x_norm * range_x
			y = int(np.clip(y, min_y, max_y))
			x = int(np.clip(x, min_x, max_x))
			pygame.draw.rect(pygame.display.get_surface(), Color(255, 255, 255), Rect(y-crop_height//2, x-crop_width//2, crop_height, crop_width), 5)
		pygame.display.update()

		x_t = preprocess(x_t, look_targets[thread_id])


		with graph.as_default():
			critic_reward = model.predict([s_t, action_state, DUMMY_ADVANTAGE, DUMMY_OLD_PRED])[1]

		# y = 0 if a_t[0] == 1 else 1
		# y = np.hstack((y, look_action))
		# y = np.reshape(y, (1, -1))

		action_and_look = np.append(actions[0:1], look_targets[thread_id])
		action_and_look = action_and_look.reshape(1, -1)
		actions = np.reshape(actions, (1, -1))
		out = np.reshape(out, (1, -1))
		critic_reward = np.reshape(critic_reward, (1, -1))

		r_store = np.append(r_store, [[r_t] * 1], axis = 0)
		state_store = np.append(state_store, s_t, axis = 0)
		action_state_store = np.append(action_state_store, action_state, axis = 0)
		action_store = np.append(action_store, actions, axis=0)
		pred_store = np.append(pred_store, out, axis = 0)
		critic_store = np.append(critic_store, critic_reward, axis=0)
		
		s_t = np.append(x_t, s_t[:, :, :, :-NUM_CROPS], axis=3)
		action_state = np.append(action_and_look, action_state[:, :-NUM_ACTIONS], axis=-1)
		print("Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Action = " + str(a_t) + ", " + str(actions) + ", Output = "+ str(out))
	
	if terminal == False:
		r_store[len(r_store)-1] = critic_store[len(r_store)-1]
	else:
		r_store[len(r_store)-1] = [-1] * 1
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
		action_state = np.zeros((1, NUM_ACTIONS * TIME_SLICES))
		look_targets[thread_id] = np.array((0., 0.))
	
	for i in range(2,len(r_store)+1):
		r_store[len(r_store)-i] = r_store[len(r_store)-i] + GAMMA*r_store[len(r_store)-i + 1]

	return s_t, action_state, state_store, action_state_store, action_store, pred_store, r_store, critic_store

#function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def step_decay(epoch):
	decay = LEARNING_RATE / 20000.
	lrate = LEARNING_RATE - epoch*decay
	lrate = max(lrate, 0.)
	return lrate

class actorthread(threading.Thread):
	def __init__(self,thread_id, s_t, action_state):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.next_state = s_t
		self.next_action_state = action_state

	def run(self):
		global episode_action
		global episode_pred
		global episode_r
		global episode_critic
		global episode_state
		global episode_action_state

		threadLock.acquire()
		self.next_state, self.next_action_state, state_store, action_state_store, action_store, pred_store, r_store, critic_store = runprocess(self.thread_id, self.next_state, self.next_action_state)
		self.next_state = self.next_state.reshape(self.next_state.shape[1], self.next_state.shape[2], self.next_state.shape[3])
		self.next_action_state = self.next_action_state.reshape(self.next_action_state.shape[1])

		episode_r = np.append(episode_r, r_store, axis = 0)
		episode_pred = np.append(episode_pred, pred_store, axis = 0)
		episode_action = np.append(episode_action, action_store, axis = 0)
		episode_state = np.append(episode_state, state_store, axis = 0)
		episode_action_state = np.append(episode_action_state, action_state_store, axis = 0)
		episode_critic = np.append(episode_critic, critic_store, axis = 0)

		threadLock.release()

states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
action_states = np.zeros((0, NUM_ACTIONS * TIME_SLICES))

#initializing state of each thread
for i in range(0, len(game_state)):
	action = np.array([[0] * NUM_ACTIONS])
	image = game_state[i].getCurrentFrame()
	image = preprocess(image, action[1:])
	state = np.concatenate((image, image, image, image), axis=3)
	states = np.append(states, state, axis = 0)
	action = action.reshape((1, -1))
	action_state = np.concatenate([action] * TIME_SLICES, axis=-1)
	action_states = np.append(action_states, action_state, axis=0)

while True:	
	threadLock = threading.Lock()
	threads = []
	for i in range(0,THREADS):
		threads.append(actorthread(i, states[i], action_states[i]))

	states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	action_states = np.zeros((0, NUM_ACTIONS * TIME_SLICES))

	for i in range(0,THREADS):
		threads[i].start()

	#thread.join() ensures that all threads fininsh execution before proceeding further
	for i in range(0,THREADS):
		threads[i].join()

	for i in range(0,THREADS):
		state = threads[i].next_state
		action_state = threads[i].next_action_state
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+1])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+2])
		# plt.show()
		state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
		states = np.append(states, state, axis = 0)
		action_state = action_state.reshape(1, -1)
		action_states = np.append(action_states, action_state, axis = 0)

	e_mean = np.mean(episode_r)
	#advantage calculation for each action taken
	advantage = episode_r - episode_critic
	# advantage = np.reshape(advantage, (-1, 1))
	print("backpropagating")

	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [lrate]

	#backpropagation
	# history = model.fit([episode_state, episode_action_state, advantage, episode_pred], {'o_P': episode_action, 'o_V': episode_r}, callbacks = callbacks_list, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)

	episode_r = np.empty((0, 1), dtype=np.float32)
	episode_pred = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
	episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
	episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	episode_action_state = np.zeros((0, NUM_ACTIONS * TIME_SLICES))
	episode_critic = np.empty((0, 1), dtype=np.float32)

	# f = open("rewards.txt","a")
	# f.write("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Loss: " + str(history.history['loss'][-1]) + "\n")
	# f.close()
	# print("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Loss: " + str(history.history['loss'][-1]))

	# summary = tf.Summary(value=[
	# 	tf.Summary.Value(tag="reward mean", simple_value=float(e_mean)),
	# 	tf.Summary.Value(tag="total loss", simple_value=float(history.history['loss'][-1])),
	# 	tf.Summary.Value(tag="action loss", simple_value=float(history.history['o_P_loss'][-1])),
	# 	tf.Summary.Value(tag="critic loss", simple_value=float(history.history['o_V_loss'][-1])),
	# 	tf.Summary.Value(tag="max score", simple_value=float(max_score))
	# ])
	# summary_writer.add_summary(summary, EPISODE)

	# if EPISODE % (20 * EPOCHS) == 0: 
	# 	model.save("saved_models/look_abs/model_updates" +	str(EPISODE)) 
	EPISODE += EPOCHS
