# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:16:50 2022

@author: User
"""
from vr_axes_reach_env import VirtualAxesReachEnv
from vr_axes_intobox import VirtualAxesIntoBoxEnv

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
import numpy as np

folder_path = 'experiments'
TEST = 'reach_beam'
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

# init environment

'''factor = np.array([5.5, 6])
transf_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
env = VirtualAxesIntoBoxEnv(axes_range = np.array([[-50, 50], [-25, 25]]), 
                          max_episode_steps = 5,
                          laser_beam_pos_px = [500, 300], 
                          stages_init_position_mm = np.array([0, 0]),
                          transform_matrix=transf_matrix, 
                          training=True)'''

factor = np.array([5.5, 6])
transf_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
env = VirtualAxesReachEnv(distance_threshold = 0.03,
                          axes_range = np.array([[-50, 50], [-25, 25]]), 
                          max_episode_steps = 10,
                          laser_beam_pos_px = [500, 300], 
                          stages_init_position_mm = np.array([0, 0]),
                          transform_matrix=transf_matrix, 
                          training=True)  

   
env = Monitor(env, log_dir)

# Create the callback: check every 500 steps
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
                n_sampled_goal=8,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=True,
                max_episode_length=10),
            verbose=0, 
            learning_rate=0.001, 
            action_noise = action_noise, 
            learning_starts=1500,
            gamma=0.95, 
            batch_size=512, 
            policy_kwargs = policy_kwargs, 
            )

# Train the model
start = time.perf_counter()
model.learn(15000, callback=callback)
model.save(model_path)
finish = time.perf_counter()
print(f"Finished in {round(finish-start, 2)} seconds")
