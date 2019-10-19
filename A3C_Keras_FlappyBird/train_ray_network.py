import numpy as np
import sys
sys.path.append("game/")

from coord import CoordinateChannel2D

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input, Concatenate
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

from keras.layers import Lambda
from keras.backend import slice

import pygame
from pygame import Rect, Color, Vector2
import wrapped_flappy_bird as game
from ray_utils import raycast

import scipy.misc
import scipy.stats as st

import threading
import time
import math
import random

import matplotlib.pyplot as plt

GAMMA = 0.99                #discount value
BETA = 0.01                 #regularisation coefficient
IMAGE_ROWS = 85
IMAGE_COLS = 84
ZOOM = 2
NUM_CROPS = 1
TIME_SLICES = 4
EXTRA_ACTIONS = 0
NUM_NORMAL_ACTIONS = 1
NUM_ACTIONS = NUM_NORMAL_ACTIONS + EXTRA_ACTIONS
IMAGE_CHANNELS = NUM_CROPS * TIME_SLICES
LEARNING_RATE_RAY = 1e-4
LEARNING_RATE_ACTION = 1e-4
LOSS_CLIPPING = 0.1
NUM_RAYS = 5
NUM_RAY_ACTIONS = NUM_RAYS + 2
RAY_START_VEL = 10
RAY_ANGLE_VEL = 0.01 * np.pi
# LOOK_SPEED = 0.1
# TEMPERATURE = 0
# TEMP_INCR = 1e-6

EPOCHS = 3
THREADS = 16
T_MAX = 15
BATCH_SIZE = 80
T = 0
EPISODE = 0

episode_r = np.empty((0, 1), dtype=np.float32)
episode_state_ray = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_state_action = np.zeros((0, (NUM_RAYS * 3 + 2) * TIME_SLICES))
episode_action_ray = np.empty((0, NUM_RAY_ACTIONS), dtype=np.float32)
episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
episode_pred_ray = np.empty((0, NUM_RAY_ACTIONS * 2), dtype=np.float32)
episode_pred_action = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
episode_critic = np.empty((0, 1), dtype=np.float32)

DUMMY_ADVANTAGE = np.zeros((1, 1))
DUMMY_OLD_RAY_PRED = np.zeros((1, NUM_RAY_ACTIONS * 2))
DUMMY_OLD_PRED  = np.zeros((1, NUM_ACTIONS * 2))

ACTIONS = 2
a_t = np.zeros(ACTIONS)

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

def ppo_loss(advantage, old_pred, num_actions, beta):
	def loss(y_true, y_pred):
		mu_pred = y_pred[:,:num_actions]
		var_pred = y_pred[:,num_actions:]
		mu_old_pred = old_pred[:,:num_actions]
		var_old_pred = old_pred[:,num_actions:]
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
		entropy_penalty = -beta * K.mean(entropy)

		return aloss + entropy_penalty
	return loss

#loss function for critic output
# def sumofsquares(y_true, y_pred):        #critic loss
# 	return K.mean(K.square(y_pred - y_true))

#function buildmodel() to define the structure of the neural network in use 
def build_ray_model():
	print("Model building begins")

	model = Sequential()
	keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	SR = Input(shape = ((NUM_RAYS * 3 + 2) * TIME_SLICES, ), name = 'Input_Ray_State')
	h0 = CoordinateChannel2D()(S)
	h0 = Conv2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h0)
	h0 = BatchNormalization()(h0)
	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h1)
	h1 = BatchNormalization()(h1)
	h2 = Flatten()(h1)
	h2 = Concatenate()([SR, h2])
	
	h3 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	h3 = BatchNormalization()(h3)
	P_mu = Dense(NUM_RAY_ACTIONS, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)
	P_sigma = Dense(NUM_RAY_ACTIONS, activation = 'softplus', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)
	P = Concatenate(name = 'o_P')([P_mu, P_sigma])
	
	A = Input(shape = (1,), name = 'Advantage')
	O = Input(shape = (NUM_RAY_ACTIONS * 2,), name = 'Old_Prediction')
	model = Model(inputs = [S,SR,A,O], outputs = P)
	# optimizer = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	optimizer = Adam(lr = LEARNING_RATE_RAY)
	model.compile(loss = ppo_loss(A,O, NUM_RAY_ACTIONS, BETA), optimizer = optimizer)
	return model

def build_action_model():
	print("Model building begins")

	model = Sequential()
	keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = ((NUM_RAYS * 3 + 2) * TIME_SLICES, ), name = 'Input')
	h1 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (S)
	h1 = BatchNormalization()(h1)
	h2 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h1)
	h2 = BatchNormalization()(h2)

	P_mu = Dense(NUM_ACTIONS, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	P_sigma = Dense(NUM_ACTIONS, activation = 'softplus', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	P = Concatenate(name = 'o_P')([P_mu, P_sigma])
	
	A = Input(shape = (1,), name = 'Advantage')
	O = Input(shape = (NUM_ACTIONS * 2,), name = 'Old_Prediction')
	model = Model(inputs = [S,A,O], outputs = P)
	# optimizer = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	optimizer = Adam(lr = LEARNING_RATE_ACTION)
	model.compile(loss = ppo_loss(A,O, NUM_ACTIONS, BETA), optimizer = optimizer)
	return model

def build_critic_model():
	print("Model building begins")

	model = Sequential()
	keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	SI = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	SR = Input(shape = ((NUM_RAYS * 3 + 2) * TIME_SLICES, ), name = 'Input_Ray_State')

	h0 = CoordinateChannel2D()(SI)
	h0 = Conv2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h0)
	h0 = BatchNormalization()(h0)
	h1 = CoordinateChannel2D()(h0)
	h1 = Conv2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h1)
	h1 = BatchNormalization()(h1)
	h2 = Flatten()(h1)
	h2 = Concatenate()([SR, h2])

	h3 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	h3 = BatchNormalization()(h3)

	V = Dense(1, name = 'o_V', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)
	
	model = Model(inputs = [SI,SR], outputs = V)
	# optimizer = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	optimizer = Adam(lr = LEARNING_RATE_ACTION)
	model.compile(loss = 'mse', optimizer = optimizer)
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
ray_model = build_ray_model()
action_model = build_action_model()
critic_model = build_critic_model()
# model.load_weights("saved_models/model_updates10080")
ray_model._make_predict_function()
action_model._make_predict_function()
graph = tf.get_default_graph()

a_t[0] = 1 #index 0 = no flap, 1= flap
#output of network represents probability of flap

game_state = []
ray_states = []
for i in range(0,THREADS):
	game_state.append(game.GameState(30000))
	starts = np.array([game_state[i].playerx, game_state[i].playery], dtype=np.float32)
	angles = np.arange(NUM_RAYS) * np.pi / (NUM_RAYS - 1) - np.pi / 2
	ray_states.append({'starts': starts, 'angles': angles})

def random_color():
    return list(np.random.choice(range(256), size=3))

ray_colors = []
for _ in range(NUM_RAYS):
	ray_colors.append(random_color())

def runprocess(thread_id, s_t, s_r_t):
	global T
	global a_t
	global ray_model
	global action_model

	t = 0
	t_start = t
	terminal = False
	r_t = 0
	r_store = np.empty((0, 1), dtype=np.float32)
	r_store_action = np.empty((0, 1), dtype=np.float32)
	state_store_ray = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	state_store_action = np.zeros((0, (NUM_RAYS * 3 + 2) * TIME_SLICES))
	action_store_ray = np.zeros((0, NUM_RAY_ACTIONS))
	action_store = np.zeros((0, NUM_ACTIONS))
	pred_store_ray = np.zeros((0, NUM_RAY_ACTIONS * 2))
	pred_store_action = np.zeros((0, NUM_ACTIONS * 2))
	critic_store = np.zeros((0, 1))
	critic_store_action = np.zeros((0, 1))
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
	s_r_t = s_r_t.reshape(1, -1)

	while t-t_start < T_MAX and terminal == False:
		t += 1
		T += 1
		# LOOK_SPEED += TEMP_INCR
		# LOOK_SPEED = np.clip(LOOK_SPEED, 0, 0.1)

		current_frame = game_state[thread_id].getCurrentFrame()

		# Perform raycast
		hit_locs, distances = raycast(current_frame, ray_states[thread_id]['starts'], ray_states[thread_id]['angles'])
		distances /= np.sqrt(np.square(current_frame.shape[0]) + np.square(current_frame.shape[1]))
		distances = np.reshape(distances, (1, -1))

		# Visualize rays
		for i in range(NUM_RAYS):
			pygame.display.get_surface().fill(ray_colors[i], rect=Rect(hit_locs[i][0] - 5, hit_locs[i][1] - 5, 10, 10))
			pygame.display.get_surface().fill(ray_colors[i], rect=Rect(ray_states[thread_id]['starts'][0] - 5, ray_states[thread_id]['starts'][1] - 5, 10, 10))
			pygame.draw.line(pygame.display.get_surface(), ray_colors[i], Vector2(ray_states[thread_id]['starts'][0],ray_states[thread_id]['starts'][1]), Vector2(hit_locs[i][0],hit_locs[i][1]), 5)
		pygame.display.update()

		# Get predicted ray vars
		with graph.as_default():
			ray_net_out = ray_model.predict([s_t, s_r_t, DUMMY_ADVANTAGE, DUMMY_OLD_RAY_PRED])[0]
		num_actions = ray_net_out.shape[0] // 2
		mu = ray_net_out[:num_actions]
		sigma_sq = ray_net_out[num_actions:]
		eps = np.random.randn(mu.shape[0])
		ray_actions = mu + np.sqrt(sigma_sq) * eps
		# ray_starts = (ray_actions[:2] + 1) / 2 * current_frame.shape[0:2]
		# ray_angles = ray_actions[2:] * np.pi
		ray_starts_v = ray_actions[0:2]
		ray_starts_v = RAY_START_VEL * ray_starts_v
		ray_angles_v = ray_actions[2:]
		ray_angles_v = RAY_ANGLE_VEL * ray_angles_v
		ray_starts = ray_states[thread_id]['starts'] + ray_starts_v
		# ray_starts = np.array([[game_state[thread_id].playerx + 10, game_state[thread_id].playery + 10]] * NUM_RAYS)
		unclipped_ray_starts = ray_starts # Unbounded
		ray_starts = np.clip(ray_starts, 0, current_frame.shape[0:2]) # Clipped to range [0, frame_size]
		ray_states[thread_id]['starts'] = ray_starts
		ray_angles = ray_states[thread_id]['angles'] + ray_angles_v
		# ray_angles = (2 * np.pi + ray_angles) % (2 * np.pi)
		ray_states[thread_id]['angles'] = ray_angles

		# Get action prediction using raycasts as input
		with graph.as_default():
			action_net_out = action_model.predict([s_r_t, DUMMY_ADVANTAGE, DUMMY_OLD_PRED])[0]
		num_actions = action_net_out.shape[0] // 2
		mu = action_net_out[:num_actions]
		sigma_sq = action_net_out[num_actions:]
		eps = np.random.randn(mu.shape[0])
		actions = mu + np.sqrt(sigma_sq) * eps

		# no = np.random.rand()
		# a_t = [0,1] if no < actions[0] else [1,0]  #stochastic action
		a_t = [0,1] if 0.5 < actions[0] else [1,0]  #deterministic action

		x_t, r_t, terminal = game_state[thread_id].frame_step(a_t)

		# Invalid action penalty
		# valid_check = np.append(actions, ray_actions)
		# valid_check = np.append(valid_check, (unclipped_ray_starts / current_frame.shape[0:2] * 2) - 1) # Range [-1, 1]
		# valid = (-1 <= valid_check) & (valid_check <= 1)
		# invalid = ~valid
		# r_t -= 0.1 * np.mean(invalid * np.abs(valid_check))

		# # Penalize changing ray_actions between frames (raycast actions should be temporally stable)
		# # ray_actions_trajectory = np.append(action_store_ray, ray_actions.reshape(1, -1), axis=0)
		# # for i in range(1, ray_actions_trajectory.shape[0]):
		# r_t -= 0.1 * np.mean(np.square(s_r_t[0, NUM_RAYS * 5 : NUM_RAYS * 10] - s_r_t[0, :NUM_RAYS * 5]))

		x_t = preprocess(x_t)

		with graph.as_default():
			critic_reward = critic_model.predict([s_t, s_r_t])

		# y = 0 if a_t[0] == 1 else 1
		# y = np.hstack((y, look_action))
		# y = np.reshape(y, (1, -1))

		ray_actions = np.reshape(ray_actions, (1, -1))
		actions = np.reshape(actions, (1, -1))
		ray_net_out = np.reshape(ray_net_out, (1, -1))
		action_net_out = np.reshape(action_net_out, (1, -1))
		critic_reward = np.reshape(critic_reward, (1, -1))

		r_store = np.append(r_store, [[r_t] * 1], axis = 0)
		state_store_ray = np.append(state_store_ray, s_t, axis = 0)
		state_store_action = np.append(state_store_action, s_r_t, axis = 0)
		action_store_ray = np.append(action_store_ray, ray_actions, axis=0)
		action_store = np.append(action_store, actions, axis=0)
		pred_store_ray = np.append(pred_store_ray, ray_net_out, axis = 0)
		pred_store_action = np.append(pred_store_action, action_net_out, axis = 0)
		critic_store = np.append(critic_store, critic_reward, axis=0)
		
		# Update observed ray state
		ray_state_starts = (ray_states[thread_id]['starts'] / current_frame.shape[0:2] * 2) - 1
		ray_state_angles = np.concatenate([np.cos(ray_states[thread_id]['angles']).reshape(-1, 1), np.sin(ray_states[thread_id]['angles']).reshape(-1, 1)], axis=-1)
		ray_state = np.concatenate([ray_state_starts.reshape(1, -1), ray_state_angles.reshape(1, -1)], axis=-1)
		s_r_t = np.append(np.concatenate([distances, ray_state], axis=-1), s_r_t[:, :-(NUM_RAYS * 3 + 2)], axis=-1)
		s_t = np.append(x_t, s_t[:, :, :, :-NUM_CROPS], axis=3)
		# action_state = np.append(action_and_look, action_state[:, :-NUM_ACTIONS], axis=-1)
		print("Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Action = " + str(a_t) + ", " + str(actions) + ", Output = "+ str(action_net_out))
	
	if terminal == False:
		r_store[len(r_store)-1] = critic_store[len(r_store)-1]
	else:
		r_store[len(r_store)-1] = [-1] * 1
		s_t = np.concatenate([x_t] * TIME_SLICES, axis=3)
		starts = np.array([game_state[thread_id].playerx, game_state[thread_id].playery], dtype=np.float32)
		angles = np.arange(NUM_RAYS) * np.pi / (NUM_RAYS - 1) - np.pi / 2
		current_frame = game_state[thread_id].getCurrentFrame()
		_, distances = raycast(current_frame, starts, angles)
		distances /= np.sqrt(np.square(current_frame.shape[0]) + np.square(current_frame.shape[1]))
		distances = np.reshape(distances, (1, -1))
		ray_states[thread_id] = {'starts': starts, 'angles': angles}
		ray_state_starts = (ray_states[thread_id]['starts'] / current_frame.shape[0:2] * 2) - 1
		ray_state_angles = np.concatenate([np.cos(ray_states[thread_id]['angles']).reshape(-1, 1), np.sin(ray_states[thread_id]['angles']).reshape(-1, 1)], axis=-1)
		ray_state = np.concatenate([ray_state_starts.reshape(1, -1), ray_state_angles.reshape(1, -1)], axis=-1)
		s_r_t = np.concatenate([distances, ray_state] * TIME_SLICES, axis=-1)
		# action_state = np.zeros((1, NUM_ACTIONS * TIME_SLICES))
		# look_targets[thread_id] = np.array((0., 0.))
	
	for i in range(2,len(r_store)+1):
		r_store[len(r_store)-i] = r_store[len(r_store)-i] + GAMMA*r_store[len(r_store)-i + 1]

	return s_r_t, s_t, state_store_ray, state_store_action, action_store_ray, action_store, pred_store_ray, pred_store_action, r_store, critic_store

#function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def create_decay_func(start_lr):
	def step_decay(epoch):
		decay = start_lr / 20000.
		lrate = start_lr - epoch*decay
		lrate = max(lrate, 0.)
		return lrate
	return step_decay

class actorthread(threading.Thread):
	def __init__(self,thread_id, s_t, s_r_t):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.next_state = s_t
		self.next_ray_state = s_r_t

	def run(self):
		global episode_r
		global episode_pred_ray
		global episode_pred_action
		global episode_action_ray
		global episode_action
		global episode_state_ray
		global episode_state_action 
		global episode_critic

		threadLock.acquire()
		# s_r_t, s_t, state_store_ray, state_store_action, action_store_ray, action_store, pred_store_ray, pred_store_action, r_store, critic_store
		self.next_ray_state, self.next_state, state_store_ray, state_store_action, action_store_ray, action_store, pred_store_ray, pred_store_action, r_store, critic_store = runprocess(self.thread_id, self.next_state, self.next_ray_state)
		self.next_state = self.next_state.reshape(self.next_state.shape[1], self.next_state.shape[2], self.next_state.shape[3])
		self.next_ray_state = self.next_ray_state.reshape(1, -1)

		episode_r = np.append(episode_r, r_store, axis = 0)
		episode_pred_ray = np.append(episode_pred_ray, pred_store_ray, axis = 0)
		episode_pred_action = np.append(episode_pred_action, pred_store_action, axis = 0)
		episode_action_ray = np.append(episode_action_ray, action_store_ray, axis = 0)
		episode_action = np.append(episode_action, action_store, axis = 0)
		episode_state_ray = np.append(episode_state_ray, state_store_ray, axis = 0)
		episode_state_action = np.append(episode_state_action, state_store_action, axis = 0)
		episode_critic = np.append(episode_critic, critic_store, axis = 0)

		threadLock.release()

states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
states_ray = np.zeros((0, (NUM_RAYS * 3 + 2) * TIME_SLICES))

#initializing state of each thread
for i in range(0, len(game_state)):
	starts = np.array([game_state[i].playerx, game_state[i].playery], dtype=np.float32)
	angles = np.arange(NUM_RAYS) * np.pi / (NUM_RAYS - 1) - np.pi / 2

	image = game_state[i].getCurrentFrame()
	_, distances = raycast(image, starts, angles)
	distances /= np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))

	ray_state_starts = (starts / image.shape[0:2] * 2) - 1
	ray_state_angles = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
	ray_state = np.concatenate([ray_state_starts, ray_state_angles], axis=-1)
	ray_state = np.concatenate([distances.reshape(1, -1), ray_state.reshape(1, -1)], axis=-1)
	ray_state = np.concatenate([ray_state] * TIME_SLICES, axis=-1)
	states_ray = np.append(states_ray, ray_state, axis=0)

	image = preprocess(image)
	state = np.concatenate(([image] * TIME_SLICES), axis=3)
	states = np.append(states, state, axis = 0)

while True:	
	threadLock = threading.Lock()
	threads = []
	for i in range(0,THREADS):
		threads.append(actorthread(i, states[i], states_ray[i]))

	states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	states_ray = np.zeros((0, (NUM_RAYS * 3 + 2) * TIME_SLICES))

	for i in range(0,THREADS):
		threads[i].start()

	#thread.join() ensures that all threads fininsh execution before proceeding further
	for i in range(0,THREADS):
		threads[i].join()

	for i in range(0,THREADS):
		state = threads[i].next_state
		ray_state = threads[i].next_ray_state
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+1])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+2])
		# plt.show()
		state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
		states = np.append(states, state, axis = 0)
		ray_state = ray_state.reshape(1, -1)
		states_ray = np.append(states_ray, ray_state, axis = 0)

	e_mean = np.mean(episode_r)
	#advantage calculation for each action taken
	advantage = episode_r - episode_critic
	# advantage = np.reshape(advantage, (-1, 1))
	print("backpropagating")

	lrate_ray = LearningRateScheduler(create_decay_func(LEARNING_RATE_RAY))
	lrate_action = LearningRateScheduler(create_decay_func(LEARNING_RATE_ACTION))
	callbacks_list_ray = [lrate_ray]
	callbacks_list_action = [lrate_action]

	#backpropagation
	history_ray = ray_model.fit([episode_state_ray, episode_state_action, advantage, episode_pred_ray], [episode_action_ray], callbacks = callbacks_list_ray, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)
	history_action = action_model.fit([episode_state_action, advantage, episode_pred_action], [episode_action], callbacks = callbacks_list_action, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)
	history_critic = critic_model.fit([episode_state_ray, episode_state_action], [episode_r], callbacks = callbacks_list_action, epochs = EPISODE + EPOCHS, batch_size = BATCH_SIZE, initial_epoch = EPISODE)

	episode_r = np.empty((0, 1), dtype=np.float32)
	episode_state_ray = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	episode_state_action = np.zeros((0, (NUM_RAYS * 3 + 2) * TIME_SLICES))
	episode_action_ray = np.empty((0, NUM_RAY_ACTIONS), dtype=np.float32)
	episode_action = np.empty((0, NUM_ACTIONS), dtype=np.float32)
	episode_pred_ray = np.empty((0, NUM_RAY_ACTIONS * 2), dtype=np.float32)
	episode_pred_action = np.empty((0, NUM_ACTIONS * 2), dtype=np.float32)
	episode_critic = np.empty((0, 1), dtype=np.float32)

	f = open("rewards.txt","a")
	f.write("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Ray_Loss: " + str(history_ray.history['loss'][-1]) + ", Action_Loss: " + str(history_action.history['loss'][-1]) + ", Critic_Loss: " + str(history_critic.history['loss'][-1]) + "\n")
	f.close()
	print("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Ray_Loss: " + str(history_ray.history['loss'][-1]) + ", Action_Loss: " + str(history_action.history['loss'][-1]) + ", Critic_Loss: " + str(history_critic.history['loss'][-1]))

	if EPISODE % (20 * EPOCHS) == 0: 
		action_model.save("saved_models/action_model_updates" + str(EPISODE))
		ray_model.save("saved_models/ray_model_updates" + str(EPISODE))
		critic_model.save("saved_models/critic_model_update" + str(EPISODE))
	EPISODE += EPOCHS
