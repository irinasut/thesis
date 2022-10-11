# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:40:33 2022

@author: User
"""

from vr_scanner_reach_env import VirtualScannerReachEnv
import sys
import os

sys.path.append(os.path.abspath(os.path.join('../..')))
from env_simulation import *
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np 
import time

folder_path = 'experiments'
TEST = 'final'
log_dir =  os.path.join(folder_path, 'logs/', TEST)
model_path = os.path.join(folder_path, 'models/', TEST)


class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last episodes
                mean_reward = np.mean(y[-self.check_freq:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
        return True

os.makedirs(log_dir, exist_ok=True)

factor = np.array([0.02, 0.02])
translation = np.array([500, 300])
transf_matrix = np.array([[factor[0], 0,  translation[0]], [0, factor[1], translation[1]], [0, 0, 1]])

env = VirtualScannerReachEnv(distance_threshold = 0.03, 
                             laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                             max_episode_steps = 3, 
                             transf_matrix=transf_matrix, 
                             reward_type='shaped')    


# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
 
goal_selection_strategy = 'future' 
policy_kwargs = dict(net_arch=[256, 256, 256])

# Initialize the model
model = SAC('MultiInputPolicy', 
            env, 
            replay_buffer_class=HerReplayBuffer, 
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=True,
                max_episode_length=3),
            verbose=0, 
            learning_rate=0.001, 
            action_noise = action_noise, 
            learning_starts=10,
            gamma=0.95,
            train_freq = (3, 'step'),
            batch_size=512, 
            policy_kwargs = policy_kwargs, 
            transfer_model = None
            )

# Train the model
start = time.perf_counter()
model.learn(15000, callback=callback)
model.save(model_path)
finish = time.perf_counter()
print(f"Finished in {round(finish-start, 2)} seconds")
