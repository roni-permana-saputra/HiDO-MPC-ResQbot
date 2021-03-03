""" 
HiDO-MPC Main ResQbot 
author: R Saputra
2021
"""

from casadi import *
import numpy as np
import time
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
import os
sys.path.append(os.path.abspath(''))
from visualisation import simulate_robot
from HiDO_MPC_resqbot_class import MPC_single_shooting_multi_objective as MPC
from resqbot_2d_simulator import Vehicle, Casualty, TargetVehicle


def cal_dist2line(current_pose, target_pose):
    tan_theta = np.tan(target_pose[2])
    b = -1
    a = tan_theta
    c = -((a*target_pose[0])+(b*target_pose[1]))
    dist2line = (fabs((a*current_pose[0])+(b*current_pose[1])+c))/(np.sqrt((a**2)+(b**2)))
    return dist2line      # sampling time [s]

def main(V0, C0, T, N):
    # input V0 C0 T N 
    print("Start MPC !!!!!")

    # Vehicle initialisation
    v1 = Vehicle(V0[0], V0[1], V0[2])

    # Casualty initialisation
    c1 = Casualty(C0[0], C0[1], C0[2], C0[3])

    # Calculate target
    target_pose = TargetVehicle(c1)

    # MPC constraints parameters
    v_max = 1.4         # Maximun linear speed
    v_min = -v_max      # Minimum linear speed
    w_max = 1.0         # Maximum angular speed
    w_min = -w_max      # Minimum angular speed

    # Start MPC
    mpciter = 0
    sim_tim = 20 # Maximum simulation time
    t0 = 0

    tic = time.time()
    start_loop = tic

    # MPC variable initialisation
    x0 = np.transpose(np.array([v1.x , v1.y , v1.yaw, 0, 0]))   # initial posture.
    xs = np.transpose(np.array([target_pose.x, target_pose.y, target_pose.yaw, 0, 0]))    # Reference posture. 
    p = transpose(vertcat(x0 , xs))                  # set the values of the parameters vector 
    u0 = np.zeros((N,2), dtype=float)                # two control inputs 

    from numpy import linalg as LA
    err_thresh = np.sqrt(((x0[0]-xs[0])**2)+((x0[1]-xs[1])**2))
    dist2line = cal_dist2line(x0[0:3],xs[0:3])
    angle_diff = pi - fabs(fabs(x0[2]-xs[2]) - pi)

    obj_val_his = np.array([])
    
    while((err_thresh > .025 or dist2line >  .025 or angle_diff > .025) and mpciter < 100): 
        # Begin the MPC stopwatch
        tic = time.time()

        # Choose the MPC formulation
        # Formulate the MPC
        if dist2line > .025:
            mpc = MPC(T, N, v_max, v_min, w_max, w_min, 0)
            print('MPC Stage: 0')
        elif angle_diff > .025:
            mpc = MPC(T, N, v_max, v_min, w_max, w_min, 1)
            print('MPC Stage: 1')
        elif err_thresh > .025:
            mpc = MPC(T, N, v_max, v_min, w_max, w_min, 2)
            print('MPC Stage: 2')     
        
        # Calculate the solution for the MPC problem
        sol = mpc.mpc_solver(p, u0)

        # sol       ---> solution of the optimisation process
        # sol['x']  ---> optimal decision variable as a result of the optimisation
        u = reshape(sol['x'], 2, N)     # Reshape the decission variable
        u.full

        # sol['f']  ---> the optimal objection function value 
        obj_val = sol['f']
        obj_val_his = np.append(obj_val_his, obj_val)

        # End the MPC stopwatch
        toc = time.time()

        # Calculate the MPC time
        MPC_time_iter = toc-tic

        # Stack mpc processing time history
        if mpciter == 0:
            mpc_proc_time_his = np.array([MPC_time_iter])
        else:
            mpc_proc_time_his = np.concatenate((mpc_proc_time_his, 
                                                np.array([MPC_time_iter])), 
                                                axis=0)

        x0, u0, t0 = mpc.move_robot(x0, u, t0)
        p = transpose(vertcat(x0 , xs))                  # set the values of the parameters vector

        # Stack control list history
        if mpciter == 0:
            u_cl = np.array(u[:,0])
        else:
            u_cl= np.concatenate((u_cl, np.array(u[:,0])), axis=1)

        # Compute OPTIMAL solution TRAJECTORY along the horizon
        ff_value = mpc.ff(u, p)
        
        # Stack state list history
        if mpciter == 0:
            xx1 = np.reshape(ff_value,(ff_value.shape[0], 
                                       ff_value.shape[1], 1))
        else:
            xx1 = np.concatenate((xx1, np.reshape(ff_value,(ff_value.shape[0], 
                                                  ff_value.shape[1], 1))), 
                                                  axis=2)
        
        # Stack time history
        if mpciter == 0:
            t = np.array([t0])
        else:
            t = np.append(t, np.array([t0]))
        
        # Stack state history
        if mpciter == 0:
            xx = np.reshape(x0, (x0.shape[0],1))
        else:
            xx = np.concatenate((xx,x0), axis=1)
            
        # Calculate error distance point
        err_thresh = np.sqrt(((x0[0]-xs[0])**2)+((x0[1]-xs[1])**2))
        # Stack dist2point error history
        if mpciter == 0:
            dist2point_his = np.array([err_thresh])
        else:
            dist2point_his= np.concatenate((dist2point_his, np.array([err_thresh])), axis=0)

        #Calculate dist2line
        dist2line = cal_dist2line(x0,xs)
        # Stack dist2point error history
        if mpciter == 0:
            dist2line_his = np.array([dist2line])
        else:
            dist2line_his= np.concatenate((dist2line_his, np.array([dist2line])), axis=0)

        #Calculate angle_diff
        angle_diff = pi - fabs(fabs(x0[2]-xs[2]) - pi)
        # Stack angle_difft error history
        if mpciter == 0:
            angle_diff_his = np.array([angle_diff])
        else:
            angle_diff_his= np.concatenate((angle_diff_his, np.array([angle_diff])), axis=0)

        mpciter = mpciter + 1
           
        print('Total ', mpciter, ' from 800')
    end_loop = time.time()

    t = np.append(t, np.array([t0]))
    xx = np.concatenate((xx,x0), axis=1)
    u_cl= np.concatenate((u_cl, np.array(u[:,0])), axis=1)
    main_loop_time = end_loop - start_loop
    ss_error = LA.norm((x0-xs),2)
    average_mpc_time = main_loop_time/(mpciter+1)
    print('average_mpc_time: ', average_mpc_time) 
    print('total_mpc_time: ', main_loop_time) 
    print("Done !!!!!")

    return t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his

    

if __name__ == '__main__':

    # MPC Parameters
    T = .25      # sampling time [s]
    N = 20     # prediction horizon

    # # Initialisation
    # V0 = np.array([0.5, -0.0, np.deg2rad(0)])
    # C0 = np.array([3.2, 0.4, 3.7, 0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_1.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t1 = mpc_proc_time_his

    # V0 = np.array([0.5, -0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, 0.4, 3.7, 0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_2.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t2 = mpc_proc_time_his

    # V0 = np.array([0.4, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, 0.4, 3.7, 0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_3.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t3 = mpc_proc_time_his

    # V0 = np.array([0.5, -0.0, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.4, 3.7, -0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_4.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t4 = mpc_proc_time_his

    # V0 = np.array([0.4, -0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.4, 3.7, -0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_5.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t5 = mpc_proc_time_his

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.4, 3.7, -0.5])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_6.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t6 = mpc_proc_time_his

    # V0 = np.array([0.5, -0.0, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_7.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t7 = mpc_proc_time_his

    # V0 = np.array([0.5, -0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_8.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t8 = mpc_proc_time_his

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t9 = mpc_proc_time_his


    V0 = np.array([0.4, -0.5, np.deg2rad(0)])
    C0 = np.array([3.2, -0.4, 3.7, -0.5])
    t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    simulate_robot(t, xx, xx1, u_cl, xs, N, T,c1)
    # show_robot_trajectory(t, xx, c1)
    # plot_control(t,xx)
    # np.savez("hido_simulation_8.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # 

    # ========================
    # TESTING PROCESSING TIME
    # ========================

    # T = .15      # sampling time [s]
    # N = 15     # prediction horizon

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t1 = mpc_proc_time_his

    # T = .15      # sampling time [s]
    # N = 20     # prediction horizon

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t2 = mpc_proc_time_his

    # T = .15      # sampling time [s]
    # N = 25     # prediction horizon

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t3 = mpc_proc_time_his

    # T = .15      # sampling time [s]
    # N = 30     # prediction horizon

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t4 = mpc_proc_time_his

    # T = .15      # sampling time [s]
    # N = 35     # prediction horizon

    # V0 = np.array([0.5, 0.5, np.deg2rad(0)])
    # C0 = np.array([3.2, -0.0, 3.7, -0.0])
    # t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his = main(V0, C0, T, N)
    # # np.savez("hido_simulation_9.npz", t, xx, xx1, u_cl, xs, c1, dist2point_his, dist2line_his, angle_diff_his, mpc_proc_time_his)
    # t5 = mpc_proc_time_his

    # print np.average(t1),np.average(t2),np.average(t3),np.average(t4),np.average(t5)

    # np.savez('hido_mpc_proc_time.npz', t1,t2,t3,t4,t5)



