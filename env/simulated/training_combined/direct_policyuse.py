# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 21:38:11 2022

@author: User
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../')))
from env_simulation import *
from vr_complex_machine_reach import VirtualComplexMachineReachEnv
from training_axes.vr_axes_reach_env import VirtualAxesReachEnv
from training_scanner.manual_transform.vr_scanner_reach_env import VirtualScannerReachEnv

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
import os
import numpy as np
import time

def rescale_goal( goal):
    low = np.array([0, 0])
    high = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]])
    return low + (0.5 * (goal + 1.0) * (high - low))

# ---- TRANSFORM DETAILS
factor_axes = np.array([5.7, 6.4])
factor_scanner = np.array([0.016, 0.016])
translation_scanner = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
transf_matrix_scan = np.array([[factor_scanner[0], 0,  translation_scanner[0]], \
                          [0, factor_scanner[1], translation_scanner[1]], [0, 0, 1]])
transf_matrix_axes = np.array([[factor_axes[0], 0, 0], [0, factor_axes[1], 0], [0, 0, 1]])

# ---- SCANNER REACH ENV
ScannerReachEnv = VirtualScannerReachEnv(distance_threshold = 0.03, 
                         laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                         max_episode_steps = 5, 
                         transf_matrix=transf_matrix_scan, 
                         reward_type='shaped')


# ---- STAGES REACH ENV   
StagesReachEnv = VirtualAxesReachEnv(axes_range = np.array([[-50, 50], [-25, 25]]), 
                      max_episode_steps = 5,
                      laser_beam_pos_px = [500, 300], 
                      stages_init_position_mm = np.array([0, 0]),
                      transform_matrix=transf_matrix_axes, 
                      training=True)

# -- load models

model_scan_path = '../training_scanner/manual_transform/experiments/models/09_06'
model_stages_path = '../training_axes/experiments/models/intobox_new'

model_scan = SAC.load(model_scan_path, env = ScannerReachEnv)
model_stages = SAC.load(model_stages_path, env = StagesReachEnv)

# ----- INIT COMPLEX ENV
max_episode_steps = 25
env = VirtualComplexMachineReachEnv(distance_threshold = 0.03,
                                    laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                                    axes_range = np.array([[-50, 50], [-25, 25]]), 
                                    max_episode_steps = max_episode_steps, 
                                    factor_scanner = factor_scanner,
                                    translation_scanner = translation_scanner,                                     
                                    transf_matrix_axes = transf_matrix_axes,
                                    training=False, 
                                    reward_type='shaped')

from shapely.geometry import Polygon, Point
scan_field = Polygon(SCAN_FIELD_DIM)

# ----- PREDICT
n_episodes = 5

for i_episode in range(n_episodes):
    obs = env.reset()
    env.render()
    time.sleep(2)
    done = False
    for t in range(max_episode_steps):
        if scan_field.contains(Point(rescale_goal(obs['desired_goal']))):
            action, _ = model_scan.predict(obs, deterministic=False) 
        else:
            action, _ = model_stages.predict(obs, deterministic=False) 
            
        observation, reward, done, info = env.step(action)        
        obs = observation
        
        env.render()
        time.sleep(2)
        
        if info['is_success'] == True:
            print("Episode finished after {} timesteps".format(t+1))
        if done:
            print(f"Episode finished")
            break
env.close()

