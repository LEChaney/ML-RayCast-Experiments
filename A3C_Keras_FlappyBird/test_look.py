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

IMAGE_ROWS = 85
IMAGE_COLS = 84
BETA = 0.01

#loss function for policy output
def logloss(y_true, y_pred):     #policy loss
	look_vars = y_pred[:,1:]
	num_lactions = look_vars.shape[1] // 2
	mu = look_vars[:,:num_lactions]
	sigma_sq = look_vars[:,num_lactions:]
	pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(y_true[:,1:] - mu) / (2. * sigma_sq))
	log_pdf = K.sum(K.log(pdf + K.epsilon()), axis=-1)
	return -(K.sum(K.log(y_true[:,0:1]*y_pred[:,0:1] + (1-y_true[:,0:1])*(1-y_pred[:,0:1]) + K.epsilon()), axis=-1) + log_pdf)
	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
	return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image, look_action):
	image = skimage.color.rgb2gray(image)
	y_norm, x_norm = look_action
	y = (y_norm + 1) / 2 * image.shape[0]
	x = (x_norm + 1) / 2 * image.shape[1]
	crop_rows = IMAGE_ROWS * 3
	crop_cols = IMAGE_COLS * 3
	y = int(np.clip(y, crop_rows//2, image.shape[0]-crop_rows//2))
	x = int(np.clip(x, crop_cols//2, image.shape[1]-crop_cols//2))
	image = image[y-crop_rows//2:y+crop_rows//2, x-crop_cols//2:x+crop_cols//2]
	image = skimage.transform.resize(image, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')	
	image = skimage.exposure.rescale_intensity(image, in_range=(0,1), out_range=(0,255))
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	return image

model = load_model("saved_models/model_updates50", custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
game_state = game.GameState(30)

currentScore = 0
topScore = 0
a_t = [1,0]
look_action = np.array((0, 0.25))
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
		plt.imshow(x_t[0,:,:,0])
		plt.show()
		s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
	
	y = model.predict(s_t)[0][0]
	no = np.random.random()	
	no = np.random.rand()
	a_t = [0,1] if no < y[0] else [1,0]    #stochastic policy
	#a_t = [0,1] if 0.5 <y[0] else [1,0]   #deterministic policy

	look_vars = y[1:]
	num_lactions = look_vars.shape[0] // 2
	mu = look_vars[:num_lactions]
	sigma_sq = look_vars[num_lactions:]
	eps = np.random.randn(mu.shape[0])
	look_action = mu + np.sqrt(sigma_sq) * eps
	
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