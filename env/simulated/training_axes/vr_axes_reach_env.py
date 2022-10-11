import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym
from gym import spaces
from gym.utils import pyglet_rendering 
import numpy as np 
import time
import random
from shapely.geometry import Polygon, Point
from pyglet import text
from env_simulation import *


class VirtualAxesReachEnv(gym.GoalEnv):
    def __init__(self, distance_threshold, axes_range, max_episode_steps, laser_beam_pos_px, 
                       stages_init_position_mm, transform_matrix, training=True, reward_type='shaped'):
        
        self.axes_range = axes_range
        self.training = training
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold 
        self.max_episode_steps = max_episode_steps 
        self.transform_matrix = transform_matrix
        self.laser_beam_pos_px = np.array(laser_beam_pos_px)
        self.stages_init_position_mm = stages_init_position_mm
        self.sim = None
        
        self.cam_view_polygon = Polygon(CAM_VIEW_DIM)
        # -- initialize the spaces
        self.action_space = spaces.Box(low = np.array([0, -1, -1]), high = np.array([1, 1, 1]), dtype=np.float32)# stage_x, stage_y (absolute movements)           
        self.observation_space = spaces.Dict({
             'observation': spaces.Box(-1., 1., shape=(2,),  dtype='float32'), # current stage_x and stage_y positions 
             'achieved_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32'),# the dot where the laser beam is pointing now
             'desired_goal': spaces.Box(-1., 1., shape=(2,), dtype='float32')  # current black dot position on the image
            })
        
        self.current_beam_pos = self.laser_beam_pos_px.copy()        
        
        
    def step(self, action):
        self.current_step += 1
        
        # -- take an action and move the stages
        previous_axes_pos = self._rescale_action(self.current_axes_position) # in real stages coordinates
        rescaled_action = self._rescale_action(action[1:3])
        self.current_axes_position = action[1:3] # scaled to [-1, 1]
        
        ''' TRANFORMATIONS '''
        ''' compute where on the image the laser beam is pointing after the stages performed a move '''
        delta_move = np.array(rescaled_action) - previous_axes_pos
        self.delta_px = (np.dot(np.append(delta_move, 1), self.transform_matrix))[:-1]
        self.delta_px = np.array(list(map(int, self.delta_px)))

        # -- laser beam movement would be inverse from the real dot on the object 
        self.current_beam_pos -= self.delta_px
        self.moved_goal += self.delta_px
        self.camview_coords -= self.delta_px
    
        obs = self._get_obs()
        
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        }

        if self.current_step >= self.max_episode_steps or info['is_success'] \
             or not self._check_if_cam_view_is_inside_of_window() \
             or not self._check_if_goal_is_inside_of_cam_view():
            done = True
            
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
       
        return obs, reward, done, info
    
    
    def _check_if_goal_is_inside_of_cam_view(self):
        # -- compute the new dot position
        poly = Polygon(self.camview_coords)
        if poly.contains(Point(self.moved_goal)):
            return True
        return False
        
        
    def _check_if_cam_view_is_inside_of_window(self):
        # -- compute the new dot position 
        if np.all(self.camview_coords[:] >= 0) and np.all( self.camview_coords[:, 0] <= WINDOW_SIZE[0]) \
            and np.all(self.camview_coords[:, 1] <= WINDOW_SIZE[1]):
             return True
        return False
        
    
    def _get_obs(self):
        desired_goal_scaled = self._scale_dot(self.goal.copy())
        achieved_goal_scaled = self._scale_dot(self.current_beam_pos.copy())
        return {
            'observation': [0, 0],      
            'achieved_goal': achieved_goal_scaled,         
            'desired_goal': desired_goal_scaled                    
        }      
        
        
    def compute_reward(self, achieved_goal, goal, info):        
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'shaped':
            return -d
        else:
            return (d < self.distance_threshold).astype(np.float)
    
    
    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold)
        
    
    # start up function prior every episode
    def reset(self):
        self.current_step = 0   
        # -- move stages to the initial position
        self.current_axes_position = np.array([0, 0]) # init axes stages 
        self.current_beam_pos = self.laser_beam_pos_px.copy()
        self.camview_coords = CAM_VIEW_DIM.copy()

        # -- sample the goal
        self.goal = self._sample_goal().copy() # sample random dot position in pixels
        self.moved_goal = self.goal.copy()
        self.current = self.moved_goal.copy()
        obs = self._get_obs()

        return obs   
  
    
    def render(self, mode='human'):
        
        if self.sim == None:
            # -- initialize the vizualization
            self.sim = Simulation()
            self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])            
            
        else:
            # -- TRAINING
            if self.training:
                # -- IF RESET
                if self.current_step == 0:                
                    self.sim.cam_view.v = CAM_VIEW_DIM.copy()
                    self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1]) 
                else:
                    # -- IF STEP
                    self.sim.cam_view.v -= self.delta_px
                    self.sim.black_dot_transform.set_translation(self.moved_goal[0], self.moved_goal[1])
                   
                self.sim.goal_dot_transform.set_translation(self.goal[0], self.goal[1]) 
                self.sim.laser_dot_transform.set_translation(self.current_beam_pos[0], self.current_beam_pos[1])  
            else: 
                # -- IF RESET
                if self.current_step == 0:
                  
                    self.sim.black_dot_transform.set_translation(self.goal[0], self.goal[1])
                    self.sim.workpiece.v = AXES_RANGE_DIM.copy()
                    self.sim.viewer.render(return_rgb_array=mode == "rgb_array")
                else:
                    # -- TESTING 
                    # -- compute trajectory
                    x_trajectory, y_trajectory = self._compute_trajectory(self.current, self.moved_goal)
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
                    self.current = self.moved_goal
        return self.sim.viewer.render(return_rgb_array=mode == "rgb_array")
    
    
    def _compute_trajectory(self, initial, destination):
        x = np.array([initial[0], destination[0]])
        y = np.array([initial[1], destination[1]])
        
        coeff =  np.polyfit(x, y, 1)
        polynomial = np.poly1d(coeff)
        
        x_trajectory = np.linspace(initial[0], destination[0], 100)
        y_trajectory = polynomial(x_trajectory)
        
        return x_trajectory, y_trajectory
        
        
    def _rescale_action(self, action): # -- to the real stages movements 
        x_min = self.axes_range[0][0]
        x_max = self.axes_range[0][1]
        
        y_min = self.axes_range[1][0]
        y_max = self.axes_range[1][1]        
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        return low + (0.5 * (action + 1.0) * (high - low))
    
    
    def _scale_action(self, action_in_mm): # -- to [-1, 1] action space
        x_min = self.axes_range[0][0]
        x_max = self.axes_range[0][1]
        
        y_min = self.axes_range[1][0]
        y_max = self.axes_range[1][1] 
        
        low=np.array([x_min, y_min])
        high=np.array([x_max, y_max]) 
        return 2.0 * ((action_in_mm - low) / (high - low)) - 1.0       
    
    
    def _scale_dot(self, goal): # -- to [-1, 1]
        low = np.array([0, 0])
        high = np.array([WINDOW_SIZE[0], WINDOW_SIZE[1]])
        return 2.0 * ((goal - low) / (high - low)) - 1.0
    
    
    def _sample_goal(self):
        num = 0
        minx, miny, maxx, maxy = self.cam_view_polygon.bounds 
        while num == 0:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if self.cam_view_polygon.contains(pnt):
                num+=1
        return np.round((pnt.coords[:])[0])
    
    
    def close(self):
        if self.sim:
            self.sim.viewer.close()
            self.sim = None


