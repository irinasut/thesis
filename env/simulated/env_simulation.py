# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:04:49 2022

@author: User
"""
import numpy as np

from gym.utils import pyglet_rendering 


CAM_VIEW_COLOR = np.array([0.93, 0.87, 0.8])
cam_sign = {
             'body': np.array([(490, 680), (490, 700), (510, 700), (510, 680)]),
             'objective_lines': np.array([(500, 680), (490, 670), (500, 680), (510, 670)])
           }

'''cam_sign = {
             'body': np.array([(565, 680), (565, 700), (585, 700), (585, 680)]),
             'objective_lines': np.array([(575, 680), (565, 670), (575, 680), (585, 670)])
           }'''

LASER_DOT_RAD = 7
BLACK_DOT_RAD = 5
LASER_DOT_COLOR = np.array([255, 0, 0])
WINDOW_SIZE = np.array([1200, 700])

# -- size of the area of the axes movements = size of a workpiece
AXES_RANGE_DIM = np.array([(50, 50), (50, 650), (1150, 650), (1150, 50)], dtype=float)
# -- size and location of the camera view
#CAM_VIEW_DIM = np.array([(240, 140), (340, 460), (710, 460), (810, 140)])
#SCAN_FIELD_DIM = np.array([(420, 220), (580, 220), (580, 380), (420, 380)])
CAM_VIEW_DIM = np.array([(300, 200), (400, 500), (750, 500), (850, 200)])
SCAN_FIELD_DIM = np.array([(400, 200), (600, 200), (600, 400), (400, 400)])

LASER_BEAM_INIT = np.array([500, 300]) 
AXES_POS_INIT = np.array([0, 0])


class Simulation():
    def __init__(self, exp=0):
        self.exp = exp
        self._init_viewer()
    
    
    def _init_viewer(self, mode='human'):
        screen_width = WINDOW_SIZE[0]
        screen_height = WINDOW_SIZE[1]
        # -- create a window 
        self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
        # -- create a workpiece of axes range
        self.workpiece = pyglet_rendering.make_polygon(AXES_RANGE_DIM.copy(), False)
        self.workpiece_transform = pyglet_rendering.Transform()
        self.workpiece.add_attr(self.workpiece_transform)
        self.viewer.add_geom(self.workpiece)
        
        # -- create a camera sign
        self.camera = pyglet_rendering.make_polygon(cam_sign['body'], True)
        self.lines = pyglet_rendering.make_polyline(cam_sign['objective_lines'])
        self.viewer.add_geom(self.camera)
        self.viewer.add_geom(self.lines)
        
        # -- create a polygon of a camera perspective view
        self.cam_view = pyglet_rendering.make_polygon(CAM_VIEW_DIM.copy(), True)          
        self.cam_view.set_color(0.93, 0.87, 0.8)  
        self.cam_view_transform = pyglet_rendering.Transform()
        self.cam_view.add_attr(self.cam_view_transform)
        self.viewer.add_geom(self.cam_view)
        
        # -- create a laser dot 
        self.laser_dot = pyglet_rendering.make_circle(LASER_DOT_RAD)
        self.laser_dot.set_color(255, 0, 0)
        self.laser_dot_transform = pyglet_rendering.Transform()
        self.laser_dot.add_attr(self.laser_dot_transform) 
        self.laser_dot_transform.set_translation(LASER_BEAM_INIT[0], LASER_BEAM_INIT[1])
        # -- initial laser beam position 
        self.viewer.add_geom(self.laser_dot)
        
        # -- create a moving black goal dot
        self.black_dot = pyglet_rendering.make_circle(BLACK_DOT_RAD)
        self.black_dot_transform = pyglet_rendering.Transform()
        self.black_dot.add_attr(self.black_dot_transform)
        self.viewer.add_geom(self.black_dot)
        
        # -- create a static grey dot as th goal we suppose to achieve
        self.goal_dot = pyglet_rendering.make_circle(BLACK_DOT_RAD)
        self.goal_dot.set_color(0.75, 0.75, 0.75)  
        self.goal_dot_transform = pyglet_rendering.Transform()
        self.goal_dot.add_attr(self.goal_dot_transform)
        self.viewer.add_geom(self.goal_dot)
        
        if self.exp == 1:
            # -- create the scanner fiels range 
            self.scanner_field = pyglet_rendering.make_polygon(SCAN_FIELD_DIM.copy(), False)
            self.scanner_field.set_color(255, 0,  0)
            self.scanner_field_transform = pyglet_rendering.Transform()
            self.scanner_field.add_attr(self.scanner_field_transform)
            self.viewer.add_geom(self.scanner_field)
        
        #self.viewer.render(return_rgb_array=mode == "rgb_array")
#sim = Simulation(1)