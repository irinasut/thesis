# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:22:13 2022

@author: User
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../')))
from env_simulation import *
from vr_complex_machine_reach import VirtualComplexMachineReachEnv
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
import os
import numpy as np
import time


folder_path = 'experiments'
TEST = 'T_reach_beam'
log_dir =  os.path.join(folder_path, 'logs/', TEST)
model_path = os.path.join(folder_path, 'models/', TEST)

factor_axes = np.array([5.5, 6])
transf_matrix_axes = np.array([[factor_axes[0], 0, 0], [0, factor_axes[1], 0], [0, 0, 1]])

factor_scanner = np.array([0.02, 0.02])
translation_scanner = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
max_episode_steps=10

env = VirtualComplexMachineReachEnv(distance_threshold = 0.03,
                                    laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                                    axes_range = np.array([[-50, 50], [-25, 25]]), 
                                    max_episode_steps = 10, 
                                    factor_scanner = factor_scanner,
                                    translation_scanner = translation_scanner,                                     
                                    transf_matrix_axes = transf_matrix_axes,
                                    training=False, 
                                    reward_type='shaped')


model = SAC.load(os.path.join(model_path), env=env)

n_episodes = 1
scanner_actions = 0
stages_actions = 0
env.reset()
env.render()
time.sleep(1)
for i_episode in range(1):
    print('________________________________________')
    obs = env.reset()
    env.render()
    time.sleep(1)
    done = False
    num = 0
    for t in range(max_episode_steps):
        print(obs)
        action, _ = model.predict(obs, deterministic=False) 
        if -1 <= action[0] <= 0:
            scanner_actions+=1
        else:
            stages_actions +=1        
        observation, reward, done, info = env.step(action)
        obs = observation
        print(reward)
        env.render()
        time.sleep(1)
        if info['is_success'] == True:
            num += 1
            #print("Episode finished after {} timesteps".format(t+1))
        if done:
            #print(f"Episode finished win {num} success")
            break
print(f'Nuber of successes is {num} out of {n_episodes}')
env.close()
print(f'how many scanner actions {scanner_actions}')
print(f'stages actions {stages_actions}')