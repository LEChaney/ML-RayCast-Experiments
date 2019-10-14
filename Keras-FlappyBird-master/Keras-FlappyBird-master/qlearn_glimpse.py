#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

import matplotlib.pyplot as plt

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2+2*3+2*3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

NUM_CROPS = 3
img_rows , img_cols = 28, 28
#Convert image into Black and white
TIME_SLICES = 4 #We stack 4 frames
img_channels = TIME_SLICES * NUM_CROPS

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


#function to preprocess an image before giving as input to the neural network
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
	
	images = np.empty((1, img_rows, img_cols, NUM_CROPS))
	for i in range(0, NUM_CROPS):
		max_dim = max(image.shape[0], image.shape[1])
		img = crop(image, look_action, max_dim // (2**i), max_dim // (2**i))
		img = skimage.transform.resize(img, (img_rows, img_cols), mode = 'constant')
		img = skimage.exposure.rescale_intensity(img, in_range=(0,1), out_range=(-1,1))
		img = img.reshape(1, img.shape[0], img.shape[1])
		images[:,:,:,i] = img

	return images


def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = preprocess(x_t, np.array((0,0)))

    s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=-1)
    #print (s_t.shape)

    #In Keras, need to reshape
    # s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    max_score = 0
    while (True):
        loss = 0
        Q_sa = 0
        max_Q_sa = 0
        action_index = [0, 0, 0, 0, 0]
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = [random.randrange(2), random.randrange(2, 5), random.randrange(5,8), random.randrange(8,11), random.randrange(11, 14)]
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)[0]       #input a stack of 4 images, get the prediction
                q_a = q[0:2]
                q_glimpse_y_1 = q[2:5]
                q_glimpse_y_2 = q[5:8]
                q_glimpse_x_1 = q[8:11]
                q_glimpse_x_2 = q[11:14]
                max_Q_a =   0  + np.argmax(q_a)
                max_Q_gy1 = 2  + np.argmax(q_glimpse_y_1)
                max_Q_gy2 = 5  + np.argmax(q_glimpse_y_2)
                max_Q_gx1 = 8  + np.argmax(q_glimpse_x_1)
                max_Q_gx2 = 11 + np.argmax(q_glimpse_x_2)
                action_index = np.array((max_Q_a, max_Q_gy1, max_Q_gy2, max_Q_gx1, max_Q_gx2))
                a_t[action_index] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t[0:2])

        y = np.sum(a_t[2:5]  * np.array((-1/3, 0, 1/3))) + np.sum(a_t[5:8]   * np.array((-1/9, 0, 1/9)))
        x = np.sum(a_t[8:11] * np.array((-1/3, 0, 1/3))) + np.sum(a_t[11:14] * np.array((-1/9, 0, 1/9)))
        x_t1 = preprocess(x_t1_colored, np.array((y, x)))

        # plt.imshow(x_t1[0, :, :, 0])
        # plt.show()
        # plt.imshow(x_t1[0, :, :, 1])
        # plt.show()
        # plt.imshow(x_t1[0, :, :, 2])
        # plt.show()

        s_t1 = np.append(x_t1, s_t[:, :, :, :img_channels-NUM_CROPS], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t, axis=0)
            state_t1 = np.concatenate(state_t1, axis=0)
            reward_t = np.hstack(reward_t).reshape([-1, 1])
            action_t = np.array(action_t).reshape([BATCH, -1])
            terminal = np.array(terminal).reshape([BATCH, -1])
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            Q_sa_a = Q_sa[:, 0:2]
            Q_sa_gy1 = Q_sa[:, 2:5]
            Q_sa_gy2 = Q_sa[:, 5:8]
            Q_sa_gx1 = Q_sa[:, 8:11]
            Q_sa_gx2 = Q_sa[:, 11:14]
            max_Q_sa_a = np.max(Q_sa_a, axis=1, keepdims=True)
            max_Q_sa_gy1 = np.max(Q_sa_gy1, axis=1, keepdims=True)
            max_Q_sa_gy2 = np.max(Q_sa_gy2, axis=1, keepdims=True)
            max_Q_sa_gx1 = np.max(Q_sa_gx1, axis=1, keepdims=True)
            max_Q_sa_gx2 = np.max(Q_sa_gx2, axis=1, keepdims=True)
            max_Q_sa = np.concatenate([max_Q_sa_a, max_Q_sa_gy1, max_Q_sa_gy2, max_Q_sa_gx1, max_Q_sa_gx2], axis=1)
            targets[np.arange(BATCH).reshape([BATCH, -1]), action_t] = reward_t + GAMMA * max_Q_sa * np.invert(terminal)

            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        max_score = max(max_score, game_state.score)

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.mean(max_Q_sa), "/ Loss ", loss, "/ Score ", max_score)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
