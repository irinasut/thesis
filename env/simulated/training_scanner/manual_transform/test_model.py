# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:24:21 2022

@author: User
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..')))
from env_simulation import *
from vr_scanner_reach_env import VirtualScannerReachEnv
from stable_baselines3 import HER, SAC
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import time
import matplotlib.pyplot as plt


folder_path = 'experiments'
TEST = 'new_env_prql'
log_dir =  os.path.join(folder_path, 'logs/', TEST)
model_path = os.path.join(folder_path, 'models/', TEST)

max_episode_steps = 5
factor = np.array([0.016, 0.016])
translation = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
transf_matrix = np.array([[factor[0], 0,  translation[0]], [0, factor[1], translation[1]], [0, 0, 1]])

env = VirtualScannerReachEnv(distance_threshold = 0.03, 
                             laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                             max_episode_steps = max_episode_steps, 
                             transf_matrix=transf_matrix, 
                             reward_type='shaped')    

model = SAC.load(model_path, env=env)
n_episodes = 1
time.sleep(1)
for i_episode in range(n_episodes):
    time.sleep(1)
    obs = env.reset()
    env.render()
    time.sleep(3)
    done = False
    num = 0
    for t in range(max_episode_steps):
        action, _ = model.predict(obs, deterministic=False)
        print(action)
        observation, reward, done, info = env.step(action)
        print(f'reward is {reward}')
        obs = observation
        env.render()
        time.sleep(1)
        if info['is_success'] == True:
            num += 1
            print("Episode finished after {} timesteps".format(t+1))
        if done:
            print(f"Episode finished win {num} success")
            break
print(f'Nuber of successes is {num} out of {n_episodes}')
env.close()
