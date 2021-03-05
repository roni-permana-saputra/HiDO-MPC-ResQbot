
""" 
HiDO-MPC ResQbot Main Executable File 
author: R Saputra
2021
"""

import time
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('corridor_experiments'))
from corridor_experiments import HiDO_MPC_resqbot_main as HIDO
from corridor_experiments.visualisation import simulate_robot

METHOD = ['HIDO', 'SMPC', 'BMPC', 'VMPC']

EXP_SETUP ={'1':[np.array([0.0, 0.0, np.deg2rad(0)]),np.array([3.2, 0.0, 3.7, 0.0])], 
            '2':[np.array([0.5, -0.5, np.deg2rad(0)]),np.array([3.2, 0.0, 3.7, 0.0])],
            '3':[np.array([0.5, 0.5, np.deg2rad(0)]),np.array([3.2, 0.0, 3.7, 0.0])],
            '4':[np.array([0.0, 0.0, np.deg2rad(0)]),np.array([3.2, -0.4, 3.7, -0.5])],
            '5':[np.array([0.5, -0.5, np.deg2rad(0)]),np.array([3.2, -0.4, 3.7, -0.5])],
            '6':[np.array([0.5, 0.5, np.deg2rad(0)]),np.array([3.2, -0.4, 3.7, -0.5])],
            '7':[np.array([0.0, 0.0, np.deg2rad(0)]),np.array([3.2, 0.4, 3.7, 0.5])],
            '8':[np.array([0.5, -0.5, np.deg2rad(0)]),np.array([3.2, 0.4, 3.7, 0.5])],
            '9':[np.array([0.5, 0.5, np.deg2rad(0)]),np.array([3.2, 0.4, 3.7, 0.5])]}

def set_init_goal(S):
    V0 = EXP_SETUP[S][0]
    C0 = EXP_SETUP[S][1]
    return V0,C0

if __name__ == '__main__':
    T = .25      # sampling time [s]
    N = 25       # prediction horizon
    M_list = ['HIDO', 'SMPC', 'BMPC', 'VMPC']
    S_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('Start MPC!!!')
    time.sleep(1)
    for M in M_list:
        for S in S_list:
            V0,C0 = set_init_goal(S)
            t,xx,xx1,u_cl,xs,c1,*_  = HIDO.main(V0, C0, T, N)
            simulate_robot(t, xx, xx1, u_cl, xs, N, T,c1, M, S)