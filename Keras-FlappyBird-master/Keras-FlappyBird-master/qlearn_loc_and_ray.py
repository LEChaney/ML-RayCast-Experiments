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
from flappy_bird_utils import raycast_fan

import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
RAY_ACTIONS = 2 # number of valid in game actions
PLR_ACTIONS_POS_X = 512 # number of valid pseudo-actions to choose location
PLR_ACTIONS_POS_Y = 288 # number of valid pseudo-actions to choose location
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
NUM_RAYS = 15

img_rows , img_cols = 80, 80
#Convert image into Black and white
#img_channels = 4 #We stack 4 frames
img_channels = 1 #We 1 frame - stacked ray lengths for 4 frames

def build_model_ray_len_to_q():
    print("Now we build the flap action model")
    model = Sequential()
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def build_model_img_to_player_loc():
    print("Now we build the location action model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*1
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(PLR_ACTIONS_POS_X*PLR_ACTIONS_POS_Y))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(model_ray,model_image,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    ray_start_0 = np.array([game_state.playerx, game_state.playery]) # TODO: replace with ray start state
    _, z_t = raycast_fan(x_t, ray_start_0, NUM_RAYS)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = x_t / 255.0
    
    #img_s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    img_s_t = x_t
    
    ray_s_t = np.stack((z_t, z_t, z_t, z_t), axis=0)
    
    #l_s_t = np.stack((ray_start_0, ray_start_0, ray_start_0, ray_start_0), axis=0)
    #print (s_t.shape)

    #In Keras, need to reshape
    #img_s_t = img_s_t.reshape(1, img_s_t.shape[0], img_s_t.shape[1], img_s_t.shape[2])  #1*80*80*4
    img_s_t = img_s_t.reshape(1, img_s_t.shape[0], img_s_t.shape[1], 1)
    
    #l_s_t = l_s_t.reshape(1,-1) #1 x 2*4 i.e. 1 x num player location coordinates x num frames
    
    ray_s_t = ray_s_t.reshape(1, -1)  #1 x num_rays*4

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model_ray.load_weights("model_ray.h5")
        model_image.load_weights("model_image.h5")
        adam = Adam(lr=LEARNING_RATE)
        model_ray.compile(loss='mse',optimizer=adam)
        adam = Adam(lr=LEARNING_RATE)
        model_image.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    max_score = 0
    while (True):
        ray_loss = 0
        ray_Q_sa = 0
        ray_action_index = 0
        ray_r_t = 0
        ray_a_t = np.zeros([RAY_ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                #print("----------Random Action----------")
                ray_action_index = random.randrange(RAY_ACTIONS)
                ray_a_t[ray_action_index] = 1
            else:
                ray_q = model_ray.predict(ray_s_t)       #input a stack of 4 images, get the prediction
                ray_max_Q = np.argmax(ray_q)
                ray_action_index = ray_max_Q
                ray_a_t[ray_max_Q] = 1
                
                ray_loss = 0
        
        img_Q_sa = 0
        img_action_index = 0
        img_r_t = 0
        img_a_t = np.zeros([PLR_ACTIONS_POS_X, PLR_ACTIONS_POS_Y])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                #print("----------Random Action----------")
                img_action_index = random.randrange([PLR_ACTIONS_POS_X, PLR_ACTIONS_POS_Y])
                img_a_t[img_action_index] = 1
            else:
                img_q = model_image.predict(img_s_t)       #input a stack of 4 images, get the prediction
                img_q = np.reshape(img_q, (PLR_ACTIONS_POS_X, PLR_ACTIONS_POS_Y))
                img_max_Q = np.unravel_index(np.argmax(img_q, axis=2), img_q.shape)
                img_action_index = img_max_Q
                img_a_t[img_max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, ray_r_t, terminal = game_state.frame_step(ray_a_t)
        
        #design reward for localization model
        
        #set reward for localization model equal to player score
        img_r_t = ray_r_t
        
        #set reward for localization model equal to euclidean dist from actual player position - cheating
        #img_r_t = np.linalg.norm( np.array(list(img_action_index)) - np.array([game_state.playerx, game_state.playery]) )
        
        #ray_start_t1 = np.array([game_state.playerx, game_state.playery]) # TODO: replace with ray start state
        ray_start_t1 = np.array(list(img_max_Q))
        _, z_t1 = raycast_fan(x_t1_colored, ray_start_t1, NUM_RAYS)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))


        x_t1 = x_t1 / 255.0


        z_t1 = z_t1.reshape(1, -1) # 1 x num_rays*4
        ray_s_t1 = np.append(z_t1, ray_s_t[:, :3*NUM_RAYS], axis=1)

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        #img_s_t1 = np.append(x_t1, img_s_t[:, :, :, :3], axis=3)
        img_s_t1 = x_t1
        
        # store the transition in D
        D.append((ray_s_t, ray_action_index, ray_r_t, ray_s_t1, img_s_t, img_action_index, img_r_t, img_s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            #Now we do the experience replay
            ray_state_t, ray_action_t, ray_reward_t, ray_state_t1, img_state_t, img_action_t, img_reward_t, img_state_t1, terminal = zip(*minibatch)
            
            ray_state_t = np.concatenate(ray_state_t)
            ray_state_t1 = np.concatenate(ray_state_t1)
            ray_targets = model_ray.predict(ray_state_t)
            ray_Q_sa = model_ray.predict(ray_state_t1)
            ray_targets[range(BATCH), ray_action_t] = ray_reward_t + GAMMA*np.max(ray_Q_sa, axis=1)*np.invert(terminal)
            
            ray_loss += ray_model.train_on_batch(ray_state_t, ray_targets)
            
            img_state_t = np.concatenate(img_state_t)
            img_state_t1 = np.concatenate(img_state_t1)
            img_targets = model_image.predict(img_state_t)
            img_Q_sa = model_image.predict(img_state_t1)
            img_targets[range(BATCH), img_action_t] = img_reward_t + GAMMA*np.max(img_Q_sa, axis=1)*np.invert(terminal)

            img_loss += img_model.train_on_batch(img_state_t, img_targets)
        
        img_s_t = img_s_t1
        ray_s_t = ray_s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save models")
            model_ray.save_weights("model_image.h5", overwrite=True)
            with open("model_ray.json", "w") as outfile:
                json.dump(model_ray.to_json(), outfile)
            img_model_image.save_weights("model_image.h5", overwrite=True)
            with open("model_image.json", "w") as outfile:
                json.dump(model_image.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        max_score = max(max_score, game_state.score)

        print("Ray - TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss, "/ Score ", max_score)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model_ray = build_model_ray_len_to_q()
    model_image = build_model_img_to_player_loc()
    trainNetwork(model_ray,model_image,args)

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