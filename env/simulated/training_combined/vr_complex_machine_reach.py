# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:52:20 2022

@author: User
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join('../')))
from env_simulation import *
import gym
from gym import spaces
#from gym.utils import pyglet_rendering 
import numpy as np 
from shapely.geometry import Polygon, Point
import time
import random
import logging 


class VirtualComplexMachineReachEnv(gym.GoalEnv):
    def __init__(self, distance_threshold, 
                       laser_action_range, axes_range, 
                       max_episode_steps, 
                       factor_scanner, 
                       translation_scanner, 
                       transf_matrix_axes, 
                       training = True, 
                       reward_type = 'shaped'):
        
        self.scanner_field_range = SCAN_FIELD_DIM
        self.sim = None
      
        self.laser_action_range = laser_action_range
        self.axes_range = axes_range
        
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold 
        self.max_episode_steps = max_episode_steps
       
        self.translation_scanner = translation_scanner
        self.factor_scanner = factor_scanner
        
        self.training = training 
        
        self.transf_matrix_axes = transf_matrix_axes
        
        self.cam_view_polygon = Polygon(CAM_VIEW_DIM)
        self.scan_field_polygon = Polygon(SCAN_FIELD_DIM)
        
        self.speed_stages = 0.5
        self.speed_scanner = 0.8
        
         
        self.weights = {
                'timestep_weight':   1,
                'timestep_weight_stages': 2,
                'speed_stages':      12, 
                'speed_scanner':     2, 
                'out_of_scanfield':  2,
                'out_of_camview':    10
                }
              
        # -- initialize the spaces
        self.action_space = spaces.Box(low = np.array([-1, -1, -1]), high = np.array([1, 1, 1]), dtype=np.float32)  # mirror_x, mirror_y             
        self.observation_space = spaces.Dict({
             'observation': spaces.Box(-1., 1., shape=(2,), dtype='float32'),
             'achieved_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32'), # in pixels
             'desired_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32') # in pixels
            })
        
    
    def step(self, action):
        self.current_step += 1
        done = False
        #print(f'action is {action}')
        action_type = action[0]
        if -1 <= action_type < 0:
            # -- action type is scanner
            self.action_type = 0
            self._step_scanner(action[1:3])
        else:
            if 0 <= action_type <= 1:
                # -- action type is 1
                self.action_type = 1 
                self._step_stages(action[1:3])
            else:
                done = True

        obs = self._get_obs()
           
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal'])
        }
        
        if self._is_goal_outside_camview(obs['desired_goal']):
            done = True        
                
        if self.current_step >= self.max_episode_steps or info['is_success']:
            done = True
        
        reward = self._reward(obs['achieved_goal'], obs['desired_goal'], info)
        #self.render()
        #time.sleep(0.5)
        return obs, reward, done, info
        
     
    
    def _is_goal_outside_camview(self, goal):
        # -- compute the new dot position
        if not self.cam_view_polygon.contains(Point(self._rescale_goal(goal))):          
            return True
        return False

    
    def _is_goal_outside_scanfield(self, goal):
        # -- compute the new dot position
        if not self.scan_field_polygon.contains(Point(self._rescale_goal(goal))):          
            return True
        return False
    
    
    def _is_goal_inside_scanfield(self, goal):
        # -- compute the new dot position
       
        if self.scan_field_polygon.contains(Point(self._rescale_goal(goal))):          
            return True
        return False



    def _step_scanner(self, action):
        # -- contract a transformation matrix 
        transf_matrix = np.array(
                [[self.factor_scanner[0], 0, self.translation_scanner[0]], 
                 [0, self.factor_scanner[1], self.translation_scanner[1]], [0, 0, 1]]
                )

        # get the value of movement 
        rescaled_action = self._rescale_scanner_action(action)
        # -- move
        prev_beam_pos = self._scale_goal(self.current_beam_pos)
        self.current_beam_pos = (np.dot(transf_matrix, np.append(rescaled_action, 1)))[:-1] # -- in image pixels
        new_beam = self._scale_goal(self.current_beam_pos)        
        self.rew_delta = np.linalg.norm(new_beam - prev_beam_pos, axis=-1)
        
    
    def _step_stages(self, action):
        self.out_scanfield_penalty = 0
        # -- take an action and move the stages
        previous_axes_pos = self._rescale_axes_action(self.current_axes_position) # in real stages coordinates
        rescaled_action = self._rescale_axes_action(action)
        self.current_axes_position = action # scaled to [-1, 1]
        
        ''' TRANFORMATIONS '''
        ''' compute where on the image the laser beam is pointing after the stages performed a move '''
        delta_move = np.array(rescaled_action) - previous_axes_pos
        self.delta_px = (np.dot(np.append(delta_move, 1), self.transf_matrix_axes))[:-1]
        self.delta_px = np.array(list(map(int, self.delta_px)))

        # -- laser beam movement would be inverse from the real dot on the object 
        prev_goal_pos = self._scale_goal(self.goal)
        self.prev_goal_was_in_scanfield = self._is_goal_inside_scanfield(prev_goal_pos)

        self.goal += self.delta_px
        if not self.scan_field_polygon.contains(Point(self.goal)):
            self.out_scanfield_penalty = 1.3
            
        new_goal = self._scale_goal(self.goal)        
        self.rew_delta = np.linalg.norm(new_goal - prev_goal_pos, axis=-1)

    
    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        return -d
      
     
    def _reward(self, achieved_goal, goal, info):
        is_out_scanfield = self._is_goal_outside_scanfield(goal)
        #print(f'Is the goal outsode of scanner field {is_out_scanfield}')
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        #print(f'Distance between goals is {d}')
        r = None
        
        if self._if_action_type_scanner():  
            #print('Action type is SCANNER')
            move_reward = self.rew_delta*self.weights['speed_scanner']
            #print(f'Move rewrad is {move_reward}')
            r = -d - is_out_scanfield*self.current_step*self.weights['timestep_weight'] - move_reward 
            #print(f'final reward is {r}')
            return r
             
        if self._if_action_type_stages():
            is_out_camview = self._is_goal_outside_camview(goal)
            
            #print('Action type is STAGES')
            move_reward = self.rew_delta*self.weights['speed_stages'] 
            #print(f'move reward is {move_reward}')
            move_penalty = is_out_scanfield*self.weights['out_of_scanfield'] + self.prev_goal_was_in_scanfield *self.current_step*self.weights['timestep_weight_stages']
            #print(f'move penalty {move_penalty}')
            r = - d - move_penalty - move_reward - is_out_camview*self.weights['out_of_camview']
            #print(f'final reward is {r}')
            return r
        
    
    def _if_action_type_stages(self):
        if self.action_type == 1:
            return True
        return False
    
    def _if_action_type_scanner(self):
        if self.action_type == 0:
            return True
        return False

        
    # start up function prior every episode
    def reset(self):
        self.current_step = 0
        self.action_type=None
      
        self.goal = np.array(self._sample_goal()).copy()
        # init all scanner and stages components
        self.current_beam_pos = LASER_BEAM_INIT.copy()        
        self.current_axes_position = AXES_POS_INIT.copy() # init axes stages 
                        
        # init randoring components
        self.current = self.goal.copy()
        # take observations 
        obs = self._get_obs()

        return obs   
    
    
    def _get_obs(self):
        #obs = np.concatenate((self.current_scanner_pos, self.current_stages_pos))
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
        else:
            if self.current_step == 0:
                if self.training:
                    self.sim.cam_view.v = CAM_VIEW_DIM.copy()
                    self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1]) 
                    time.sleep(1)
                    self.sim.scanner_field.v = SCAN_FIELD_DIM.copy()
                  
                # if testing
                else:
                    self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])
                    self.sim.workpiece.v = AXES_RANGE_DIM.copy()
                    self.sim.viewer.render(return_rgb_array=mode == "rgb_array")
                self.sim.laser_dot_transform.set_translation(self.current_beam_pos[0], self.current_beam_pos[1])
                return self.sim.viewer.render(return_rgb_array=mode == "rgb_array") 
            
            if self.action_type == 0:
                self.sim.laser_dot_transform.set_translation(self.current_beam_pos[0], self.current_beam_pos[1])
                self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])
            if self.action_type == 1:
                self._render_stages(mode)
            
        return self.sim.viewer.render(return_rgb_array=mode == "rgb_array") 

    
    def _render_stages(self, mode='human'):        
        # -- TRAINING
        if self.training:           
            # -- IF STEP
            self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])              
            self.sim.laser_dot_transform.set_translation(self.current_beam_pos[0], self.current_beam_pos[1])  
            
        else: 
            # -- TESTING 
            # -- compute trajectory
            x_trajectory, y_trajectory = self._compute_trajectory(self.current, self.goal)
            curr_black_dot_pos = self.current.copy()
            # -- render movement 
            for i in range(len(x_trajectory)):
                move_to = np.array([x_trajectory[i], y_trajectory[i]])
                delta_px = move_to - curr_black_dot_pos
               
                self.sim.black_dot_transform.set_translation(move_to[0], move_to[1])
                curr_black_dot_pos = move_to
                self.sim.workpiece.v += delta_px
                self.sim.viewer.render(return_rgb_array=mode == "rgb_array")
                time.sleep(0.01)
            self.current = self.goal.copy()

    
    def _compute_trajectory(self, initial, destination):
    
        x = np.array([initial[0], destination[0]])
        y = np.array([initial[1], destination[1]])
        
        coeff =  np.polyfit(x, y, 1)
        polynomial = np.poly1d(coeff)
        
        x_trajectory = np.linspace(initial[0], destination[0], 100)
        y_trajectory = polynomial(x_trajectory)
        
        return x_trajectory, y_trajectory
    
    
    def close(self):
        if self.sim:
            self.sim.viewer.close()
            self.sim = None
    
    
    def _sample_goal(self):
        num = 0
        minx, miny, maxx, maxy = self.cam_view_polygon.bounds 
        while num == 0:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if self.cam_view_polygon.contains(pnt):
                num+=1
        #return np.array([320, 350])
        #return np.array([550, 160])
        return np.array([750, 280])
        #return np.array([450, 280])
        #return np.round((pnt.coords[:])[0])
    
    '''def _sample_goal(self):
        goal = [random.uniform(self.scanner_field_range[0][0], self.scanner_field_range[1][0]), 
                random.uniform(self.scanner_field_range[0][1], self.scanner_field_range[2][1])]
        return list(map(int, goal))'''
      
    
    
    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold)
        
    
    def _rescale_scanner_action(self, action):
    
        x_min = self.laser_action_range[0][0]
        x_max = self.laser_action_range[0][1]
        
        y_min = self.laser_action_range[1][0]
        y_max = self.laser_action_range[1][1]        
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        
        return low + (0.5 * (action + 1.0) * (high - low))
    
    
    def _rescale_axes_action(self, action):
        x_min = self.axes_range[0][0]
        x_max = self.axes_range[0][1]
        
        y_min = self.axes_range[1][0]
        y_max = self.axes_range[1][1]        
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        return low + (0.5 * (action + 1.0) * (high - low))
         
      
    def _scale_goal(self, goal):
        low = np.array([0, 0])
        high = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]])
        return 2.0 * ((goal - low) / (high - low)) - 1.0


    def _rescale_goal(self, goal):
        low = np.array([0, 0])
        high = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]])
        return low + (0.5 * (goal + 1.0) * (high - low))



'''factor_axes = np.array([5.5, 6])
transf_matrix_axes = np.array([[factor_axes[0], 0, 0], [0, factor_axes[1], 0], [0, 0, 1]])

factor_scanner = np.array([0.02, 0.02])
translation_scanner = np.array([LASER_BEAM_INIT[0], LASER_BEAM_INIT[1]])


env = VirtualComplexMachineReachEnv(distance_threshold = 0.03,
                                    laser_action_range = np.array([[-5000, 5000], [-5000, 5000]]),
                                    axes_range = np.array([[-50, 50], [-25, 25]]), 
                                    max_episode_steps = 10, 
                                    factor_scanner = factor_scanner,
                                    translation_scanner = translation_scanner,                                     
                                    transf_matrix_axes = transf_matrix_axes,
                                    training=False, 
                                    reward_type='shaped')


episodes = 1
for episode in range(episodes):
    print('EPISODE ----------------------------------------------------------')
    obs = env.reset()
    print(f'Initialized episode is {obs}')
    env.render() 
    time.sleep(3)
    for _ in range (10):
        print('STEP _________________________________________________________')
        action = env.action_space.sample() # your agent here (this takes random actions)
        print(action)
        observation, reward, done, info = env.step(action)
        print(observation)
        env.render()  
        time.sleep(5)
        if done:
            print('DONE!')
            break
env.reset()
env.render()
time.sleep(1)
env.close()'''

'''episodes = 1
for episode in range(episodes):
    env.reset()
    env.render() 
    time.sleep(2)
    
    print('_________________________________________________________')
    #action = env.action_space.sample() # your agent here (this takes random actions)
    action = np.array([-0.56768, -0.3446, 0.343343], dtype=float)
    observation, reward, done, info = env.step(action)
    env.render()  
    time.sleep(3)
    if done:
        env.reset()
        print('RESET!')
        env.render() 
    time.sleep(2)
    
    print('_________________________________________________________')
    action = np.array([ 0.27962375, -0.05379102, -0.09415762], dtype=float)
    observation, reward, done, info = env.step(action)
    env.render()  
    time.sleep(3)
    if done:
        env.reset()
        print('RESET!')
        env.render() 
    time.sleep(2)
    print('_________________________________________________________')
    action = np.array([-0.56768, 0, 0], dtype=float)
    observation, reward, done, info = env.step(action)
    env.render()  
    time.sleep(3)
    env.reset()
    if done:
        print('RESET!')
        env.render() 
        time.sleep(2)
#env.reset()
#env.render()
time.sleep(1)
env.close()'''

