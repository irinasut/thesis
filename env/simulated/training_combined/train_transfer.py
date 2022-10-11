# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:13:34 2022

@author: User
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.abspath(os.path.join('../')))
from env_simulation import *
from vr_complex_machine_reach import VirtualComplexMachineReachEnv
from training_axes.vr_axes_intobox import VirtualAxesIntoBoxEnv
from training_axes.vr_axes_reach_env import VirtualAxesReachEnv

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


def train(env, filename, transfer_model = None):
    
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
        
    # Initialize the model
    model = SAC('MultiInputPolicy', 
                env, 
                verbose=0, 
                learning_rate=0.001, 
                learning_starts=1000,
                gamma=0.95, 
                train_freq = (5, 'step'),
                action_noise = action_noise, 
                batch_size=64, 
                policy_kwargs = dict(net_arch=[256, 256, 256]), 
                transfer_model = transfer_model
                )
    
    # Train the model
    start = time.perf_counter()
    model.learn(25000, callback=callback)
    model.save(model_path)
    finish = time.perf_counter()
    print(f"Finished in {round(finish-start, 2)} seconds")


if __name__ == "__main__":

    
    factor_axes = np.array([5.5, 6])
    transf_matrix_axes = np.array([[factor_axes[0], 0, 0], [0, factor_axes[1], 0], [0, 0, 1]])
    
    factor_scanner = np.array([0.02, 0.02])
    translation_scanner = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
    transf_matrix_scanner = np.array([[factor_scanner[0], 0,  translation_scanner[0]],\
                                      [0, factor_scanner[1],  translation_scanner[1]], [0, 0, 1]])

    
    env = VirtualComplexMachineReachEnv(distance_threshold = 0.03,
                                    laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                                    axes_range = np.array([[-50, 50], [-25, 25]]), 
                                    max_episode_steps = 10, 
                                    factor_scanner = factor_scanner,
                                    translation_scanner = translation_scanner,                                     
                                    transf_matrix_axes = transf_matrix_axes,
                                    training=True, 
                                    reward_type='shaped')
    
    # -- transferred env    
     
    factor_axes = np.array([5.5, 6])
    transf_matrix_axes = np.array([[factor_axes[0], 0, 0], [0, factor_axes[1], 0], [0, 0, 1]])
    
    factor_scanner = np.array([0.02, 0.02])
    translation_scanner = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])
    transf_matrix_scanner = np.array([[factor_scanner[0], 0,  translation_scanner[0]],\
                                      [0, factor_scanner[1],  translation_scanner[1]], [0, 0, 1]])
        
    env_transfer_scan = VirtualScannerReachEnv(distance_threshold = 0.03, 
                             laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                             max_episode_steps = 5, 
                             transf_matrix=transf_matrix_scanner, 
                             reward_type='shaped')
    
    env_transfer_stages = VirtualAxesIntoBoxEnv(axes_range = np.array([[-50, 50], [-25, 25]]), 
                          max_episode_steps = 5,
                          laser_beam_pos_px = [500, 300], 
                          stages_init_position_mm = np.array([0, 0]),
                          transform_matrix=transf_matrix_axes, 
                          training=True)
    
    transfer_scan_path = '../training_scanner/manual_transform/experiments/models/09_06'
    
    transfer_stages_path = '../training_axes/experiments/models/reach_beam'
    
    transfer_scan = SAC.load(transfer_scan_path, env = env_transfer_scan)
    transfer_stages = SAC.load(transfer_stages_path, env = env_transfer_stages)
   
     
    TEST = 'T_scan'
    train(env, TEST, transfer_model=transfer_scan)
    
    TEST = 'T_stages'
    train(env, TEST, transfer_model=transfer_stages)
    
    TEST = 'scratch'
    train(env, TEST, transfer_model=None)