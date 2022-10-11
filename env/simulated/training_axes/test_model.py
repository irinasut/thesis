# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:09:22 2022

@author: User
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from vr_axes_reach_env import VirtualAxesReachEnv
from vr_axes_intobox import VirtualAxesIntoBoxEnv

from stable_baselines3 import HER, SAC
import os
import numpy as np
import time
import matplotlib.pyplot as plt

max_episode_steps = 5 


'''factor = np.array([5.5, 6])
transf_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
env = VirtualAxesIntoBoxEnv(axes_range = np.array([[-50, 50], [-25, 25]]), 
                          max_episode_steps = 5,
                          laser_beam_pos_px = [500, 300], 
                          stages_init_position_mm = np.array([0, 0]),
                          transform_matrix=transf_matrix, 
                          training=False)  '''

factor = np.array([5.5, 6])
transf_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
env = VirtualAxesReachEnv(distance_threshold = 0.03,
                          axes_range = np.array([[-50, 50], [-25, 25]]), 
                          max_episode_steps = 5,
                          laser_beam_pos_px = [500, 300], 
                          stages_init_position_mm = np.array([0, 0]),
                          transform_matrix=transf_matrix, 
                          training=False)  



folder_path = 'experiments'
TEST = 'reach_beam'
log_dir =  os.path.join(folder_path, 'logs/', TEST)
model_path = os.path.join(folder_path, 'models/', TEST)
model = SAC.load(os.path.join(model_path), env=env)


n_episodes = 1
time.sleep(1)
for i_episode in range(n_episodes):
    obs = env.reset()
    print(f'obs is {obs}')
    env.render()
    time.sleep(2)
    done = False
    num = 0
    for t in range(max_episode_steps):
        action, _ = model.predict(obs, deterministic=False)
        observation, reward, done, info = env.step(action)
        print(f'observation is {observation}')
        obs = observation
        print(f'action is {action}')
        env.render()
        time.sleep(2)
        if info['is_success'] == True:
            num += 1 
            print("Episode finished after {} timesteps".format(t+1))
        if done:
            print(f"Episode finished win {num} success")
            break
print(f'Number of successes is {num} out of {n_episodes}')
env.close()

'''from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


plot_results([log_dir], 15000, results_plotter.X_TIMESTEPS, "Learning Curve SAC")
#plt.show()
plt.savefig('stages_box_curve_09.png')'''