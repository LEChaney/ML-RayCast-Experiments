import numpy as np
import sys
sys.path.append("game/")

import pygame
import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K

import matplotlib.pyplot as plt

NUM_CROPS = 3
IMAGE_CHANNELS = 4 * NUM_CROPS
IMAGE_ROWS = 28
IMAGE_COLS = 28
BETA = 0.01

#loss function for policy output
# def logloss(advantage):     #policy loss
# 	def logloss_impl(y_true, y_pred):
# 		look_vars = y_pred[:,1:]
# 		num_lactions = look_vars.shape[1] // 2
# 		mu = look_vars[:,:num_lactions]
# 		sigma_sq = look_vars[:,num_lactions:]
# 		pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(y_true[:,1:] - mu) / (2. * sigma_sq))
# 		pdf_loss = K.sum(K.log(pdf + K.epsilon()), axis=-1)
# 		act_loss = K.sum(K.log(y_true[:,0:1]*y_pred[:,0:1] + (1-y_true[:,0:1])*(1-y_pred[:,0:1]) + K.epsilon()), axis=-1)
# 		act_entropy = K.sum(y_pred[:,0:1] * K.log(y_pred[:,0:1] + K.epsilon()) + (1-y_pred[:,0:1]) * K.log(1-y_pred[:,0:1] + K.epsilon()), axis=-1)
# 		pdf_entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.), axis=-1)
# 		loss = -(pdf_loss + act_loss) * advantage - BETA * (act_entropy + pdf_entropy)
# 		return loss
# 	return logloss_impl

#loss function for critic output
# def sumofsquares(y_true, y_pred):        #critic loss
# 	return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image, look_action):
	def crop(img, look_action, crop_height, crop_width):
		crop_height = min(crop_height, img.shape[0])
		crop_width = min(crop_width, img.shape[1])
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
	
	images = np.empty((1, IMAGE_ROWS, IMAGE_COLS, NUM_CROPS))
	for i in range(0, NUM_CROPS):
		max_dim = max(image.shape[0], image.shape[1])
		img = crop(image, look_action, max_dim // (2**i), max_dim // (2**i))
		img = skimage.transform.resize(img, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')	
		img = skimage.exposure.rescale_intensity(img, in_range=(0,1), out_range=(-1,1))
		img = img.reshape(1, img.shape[0], img.shape[1])
		images[:,:,:,i] = img

	return images

model = load_model("saved_models/model_updates1800")#, custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
game_state = game.GameState(30)

currentScore = 0
topScore = 0
a_t = [1,0]
look_action = np.array((0, 0.0))
FIRST_FRAME = True

terminal = False
r_t = 0
while True:
	if FIRST_FRAME:
		x_t = game_state.getCurrentFrame()
		x_t = preprocess(x_t, look_action)
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
		FIRST_FRAME = False		
	else:
		x_t, r_t, terminal = game_state.frame_step(a_t)
		x_t = preprocess(x_t, look_action)
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+1])
		# plt.show()
		# plt.imshow(state[:, :, IMAGE_CHANNELS-NUM_CROPS+2])
		# plt.show()
		s_t = np.append(x_t, s_t[:, :, :, :IMAGE_CHANNELS-NUM_CROPS], axis=3)
	
	out = model.predict(s_t)[0][0]

	num_actions = out.shape[0] // 2
	mu = out[:num_actions]
	sigma_sq = out[num_actions:]
	eps = np.random.randn(mu.shape[0])
	actions = mu + np.sqrt(sigma_sq) * eps
	look_action = actions[1:]

	# no = np.random.rand()
	# a_t = [0,1] if no < actions[0] else [1,0]  #stochastic action
	a_t = [0,1] if 0.0 < actions[0] else [1,0]  #deterministic action
	
	if(r_t == 1):
		currentScore += 1
		topScore = max(topScore, currentScore)
		print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))
	if terminal == True:
		FIRST_FRAME = True
		terminal = False
		currentScore = 0
				


#-------------- code for checking performance of saved models by finding average scores for 10 runs------------------

# for i in range(1,6):
# 	model = load_model("model" + str(i), custom_objects={'binarycrossentropy': bce, 'sumofsquares': ss})
# 	score = 0
# 	counter = 0
# 	while counter<10:	
# 		x_t, r_t, terminal = game_state.frame_step(a_t)

# 		score += 1
# 		if r_t == -1:
# 			counter += 1
 	
# 		x_t = preprocess(x_t)

# 		if FIRST_FRAME:
# 			s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
			
# 		else:
# 			s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

# 		y = model.predict(s_t)
# 		no = np.random.random()
		
# 		print(y)
# 		if FIRST_FRAME:
# 			a_t = [0,1]
# 			FIRST_FRAME = False
# 		else:
# 			no = np.random.rand()
# 			a_t = [0,1] if no < y[0] else [1,0]	
# 			#a_t = [0,1] if 0.5 <y[0] else [1,0]
		
# 		if r_t == -1:
# 			FIRST_FRAME = True

# 	f = open("test_rewards.txt","a")
# 	f.write(str(score/10)+ "\n")
# 	f.close()
