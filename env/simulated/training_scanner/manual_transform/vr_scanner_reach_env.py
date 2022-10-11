# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:56:40 2022

@author: User
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..')))
from env_simulation import *
import gym
from gym import spaces
from gym.utils import pyglet_rendering 
import numpy as np 
import math
import time
import random
import logging 


class VirtualScannerReachEnv(gym.GoalEnv):
    def __init__(self, distance_threshold, laser_action_range, max_episode_steps, transf_matrix, reward_type = 'shaped'):
        
        self.scanner_field_range = SCAN_FIELD_DIM
        self.sim = None
      
        self.laser_action_range = laser_action_range
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold 
        self.max_episode_steps = max_episode_steps
       
        self.transf_matrix = transf_matrix
        
        # -- initialize the spaces
        self.action_space = spaces.Box(low = np.array([-1, -1, -1]), high = np.array([0, 1, 1]), dtype=np.float32)  # mirror_x, mirror_y              
        self.observation_space = spaces.Dict({
             'observation':  spaces.Box(-1., 1., shape=(2,), dtype='float32'),
             'achieved_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32'),
             'desired_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32')
            })
       
    
    def step(self, action):
        self.current_step += 1
        
        rescaled_action = self._rescale_action(action[1:3])

        self.current_scanner_pos = rescaled_action
  
        self.current_beam_pos = (np.dot(self.transf_matrix, np.append(self.current_scanner_pos, 1)))[:-1] # -- in image pixels
        
        obs = self._get_obs()
                
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }
        if self.current_step >= self.max_episode_steps or info['is_success']:
            done = True
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        #self.render()
        #time.sleep(0.5)
        return obs, reward, done, info
        
    
    def compute_reward(self, achieved_goal, goal, info):        
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'shaped':
            return -d
        else:
            return (d < self.distance_threshold).astype(np.float)
        
        
    # start up function prior every episode
    def reset(self):
        self.current_step = 0
        self.current_scanner_pos = np.array([0, 0], dtype=float)
        # sample the goal
        self.goal = np.array(self._sample_goal()).copy()
     
        #self.current_mirr_pos = np.array([0, 0])      
    
        self.current_beam_pos = LASER_BEAM_INIT.copy()
        obs = self._get_obs()
        #self.render()
        #time.sleep(0.5)
        return obs   
    
    
    def _get_obs(self):
        desired_goal_scaled = self._scale_goal(self.goal.copy())
        achieved_goal_scaled = self._scale_goal(self.current_beam_pos.copy())
        return {
            'observation': [0, 0],     
            'achieved_goal': achieved_goal_scaled,         
            'desired_goal': desired_goal_scaled                    
        }        
        
    
    def render(self, mode='human'):
        
        if self.sim  == None:
            self.sim = Simulation(1)
            self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])
            
        self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])
        self.sim.laser_dot_transform.set_translation(self.current_beam_pos[0], self.current_beam_pos[1])
        
        return self.sim.viewer.render(return_rgb_array=mode == "rgb_array")
    
    
    def close(self):
        if self.sim:
            self.sim.viewer.close()
            self.sim = None
    
    
    def _sample_goal(self):
        goal = [random.uniform(self.scanner_field_range[0][0], self.scanner_field_range[1][0]), 
                random.uniform(self.scanner_field_range[0][1], self.scanner_field_range[2][1])]
        return list(map(int, goal))
      
    
    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold)
    
    
    def _rescale_action(self, action):
        x_min = self.laser_action_range[0][0]
        x_max = self.laser_action_range[0][1]
        
        y_min = self.laser_action_range[1][0]
        y_max = self.laser_action_range[1][1]        
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        return low + (0.5 * (action + 1.0) * (high - low))


    def _scale_action(self, action):
        x_min = self.laser_action_range[0][0]
        x_max = self.laser_action_range[0][1]
        
        y_min = self.laser_action_range[1][0]
        y_max = self.laser_action_range[1][1]        
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        return 2.0 * ((action - low) / (high - low)) - 1.0
    
    
    def _scale_goal(self, goal):
        low = np.array([0, 0])
        high = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]])
        return 2.0 * ((goal - low) / (high - low)) - 1.0


"""factor = np.array([0.02, 0.02])
translation = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])

transf_matrix = np.array([[factor[0], 0,  translation[0]], [0, factor[1], translation[1]], [0, 0, 1]])

env = VirtualScannerReachEnv(distance_threshold = 0.03, 
                             laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                             max_episode_steps = 3, 
                             transf_matrix=transf_matrix, 
                             reward_type='shaped')    
episodes = 1
for episode in range(episodes):
    env.reset()
    env.render() 
    time.sleep(25)
    for _ in range (1):
        print('_________________________________________________________')
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print(observation)
        env.render()  
        time.sleep(25)
        if done:
            break
env.reset()
env.render()
time.sleep(1)
env.close()
"""