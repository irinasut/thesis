# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:48:04 2022

@author: User
"""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.abspath(os.path.join('../../')))
from env_simulation import *

from training_scanner.manual_transform.vr_scanner_reach_env import VirtualScannerReachEnv
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
import numpy as np


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


def train(env, filename, transfer_model = None, policy_library = None):
    
    folder_path = 'experiments'
    TEST = filename
    log_dir =  os.path.join(folder_path, 'logs/', TEST)    
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(folder_path, 'models/', TEST)
    
    env = Monitor(env, log_dir)
    # Create the callback: check every 500 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
  
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    
    model = SAC('MultiInputPolicy', 
               env, 
               replay_buffer_class=HerReplayBuffer, 
               replay_buffer_kwargs=dict(
                   n_sampled_goal=4,
                   goal_selection_strategy='future',
                   online_sampling=True,
                   max_episode_length=3),
               verbose=0, 
               learning_rate=0.001, 
               action_noise = action_noise, 
               learning_starts=1500,
               gamma=0.95, 
               batch_size=16, 
               policy_kwargs = dict(net_arch=[256, 256, 256]), 
               transfer_model=transfer_model, 
               policy_lib = policy_library
               )
   
    # Train the model
    start = time.perf_counter()
    model.learn(10000, callback=callback)
    model.save(model_path)
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} seconds")


if __name__ == "__main__":
    
    result_file = 'prql_scan'

    
    factor = np.array([0.016, 0.016])
    translation = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
    transf_matrix = np.array([[factor[0], 0,  translation[0]], [0, factor[1], translation[1]], [0, 0, 1]])

    env_scan = VirtualScannerReachEnv(distance_threshold = 0.03, 
                             laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                             max_episode_steps = 5, 
                             transf_matrix=transf_matrix, 
                             reward_type='shaped')
    

    transfer_scan_path = 'experiments/models/09_06'

    policy_library = []
    
    policy_library.append(SAC.load(transfer_scan_path, env=env_scan).policy)
        
    #print(f'policy library is {policy_library}')
    train(env_scan, result_file, None,  policy_library)
  
   
    