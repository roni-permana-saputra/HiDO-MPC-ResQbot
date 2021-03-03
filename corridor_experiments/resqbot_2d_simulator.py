""" 
ResQbot 2D Simulator model
    - Robot/Vehicle model 2D model
    - Casualty 2D model
author: R Saputra
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Vehicle():
    def __init__(self, ix, iy, iyaw):
        self.x = ix
        self.y = iy
        self.yaw = iyaw
        self._calc_contour()
        self.calc_global_vehicle_contour()
        
    def calc_global_vehicle_contour(self):
        self.gvx = [(ix * np.cos(self.yaw) + iy * np.sin(self.yaw)) +
                    self.x for (ix, iy) in zip(self.c_x, self.c_y)]
        self.gvy = [(ix * np.sin(self.yaw) - iy * np.cos(self.yaw)) +
                    self.y for (ix, iy) in zip(self.c_x, self.c_y)]
    
    def _calc_contour(self):
        self.c_x = [-0.3, -0.15, 0.15, 0.15, 
                    -0.15, -0.15, 0.15, 0.3, 
                    0.3, 1.35, 1.35, 0.3, 
                    0.3, 1.35, 1.35, 0.3, 
                    0.3, 1.35, 1.35, 0.3, 
                    0.3, 0.15, 0.15, -0.15, 
                    -0.15, 0.15, -0.3, -0.3]
        self.c_y = [-0.3, -0.3, -0.3, -0.35, 
                    -0.35, -0.3, -0.3, -0.3, 
                    -0.2, -0.2, -0.15, -0.15, 
                    -0.2, -0.2, 0.15, 0.15, 
                    0.2, 0.2, -0.2, -0.2, 
                    0.3, 0.3, 0.35, 0.35, 
                    0.3, 0.3, 0.3, -0.3]

class Casualty():
    def __init__(self, hx, hy, bx, by):
        self.x = hx
        self.y = hy
        # Calculate casualty orientation
        self.yaw = math.atan2((by-hy),(bx-hx))
        # Calculate the polygon
        self._calc_casualty_contour()
        self.calc_casualty_global_contour()

    def calc_casualty_global_contour(self):
        self.gcx = [(hx * np.cos(self.yaw) + hy * np.sin(self.yaw)) +
              self.x for (hx, hy) in zip(self.cc_x, self.cc_y)]
        self.gcy = [(hx * np.sin(self.yaw) - hy * np.cos(self.yaw)) +
              self.y for (hx, hy) in zip(self.cc_x, self.cc_y)]

    def _calc_casualty_contour(self):
        self.cc_x = [-0.1, -0.05, 0.15, 0.2, 
                    0.25, 0.25, 0.3, 0.8, 
                    0.8, 1.4, 1.4, 1.375, 
                    1.4, 1.4, 0.8, 0.8, 
                    0.3, 0.25, 0.25, 0.2,
                    0.15,-0.05, -0.1]
        self.cc_y = [0.0, 0.1, 0.1, 0.05, 
                     0.05, 0.2, 0.25, 0.25, 
                     0.15, 0.15, 0.05, 0.0, 
                     -0.05, -0.15, -0.15, -0.25, 
                     -0.25, -0.2, -0.05, -0.05, 
                     -0.1, -0.1, 0.0]

class TargetVehicle():
    def __init__(self, casualty):
        distance = 1.75
        self.x = casualty.x - distance*np.cos(casualty.yaw)
        self.y = casualty.y - distance*np.sin(casualty.yaw)
        self.yaw = casualty.yaw
        # Calculate the polygon
        self._calc_target_vehicle_contour()
        self.calc_global_target_contour()

    def calc_global_target_contour(self):
        self.gtx = [(tx * np.cos(self.yaw) + ty * np.sin(self.yaw)) +
                    self.x for (tx, ty) in zip(self.tc_x, self.tc_y)]
        self.gty = [(tx * np.sin(self.yaw) - ty * np.cos(self.yaw)) +
                    self.y for (tx, ty) in zip(self.tc_x, self.tc_y)]

    def _calc_target_vehicle_contour(self):
        self.tc_x = [-0.3, -0.15, 0.15, 0.15, 
                     -0.15, -0.15, 0.15, 0.3, 
                     0.3, 1.35, 1.35, 0.3, 
                     0.3, 1.35, 1.35, 0.3, 
                     0.3, 1.35, 1.35, 0.3, 
                     0.3, 0.15, 0.15, -0.15, 
                     -0.15, 0.15, -0.3, -0.3]
        self.tc_y = [-0.3, -0.3, -0.3, -0.35, 
                     -0.35, -0.3, -0.3, -0.3, 
                     -0.2, -0.2, -0.15, -0.15, 
                     -0.2, -0.2, 0.15, 0.15, 
                     0.2, 0.2, -0.2, -0.2, 
                     0.3, 0.3, 0.35, 0.35, 
                     0.3, 0.3, 0.3, -0.3]

def main():
    print("start!!")
    print("done!!")

if __name__ == '__main__':
    main()