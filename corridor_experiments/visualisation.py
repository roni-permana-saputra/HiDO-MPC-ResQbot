""" 
MPC ResQbot Visualisation 
Author: R Saputra
2021
""" 

import os
import numpy as np

import matplotlib.pyplot as plt
from resqbot_2d_simulator import Vehicle, TargetVehicle


def cal_heading_line(target_pose):
    # line define by ax + by + c = 0
    # tan_theta = -(a/b)
    tan_theta = np.tan(target_pose.yaw)
    b = -1
    a = tan_theta
    c = -((a*target_pose.x)+(b*target_pose.y))
    return a, b, c

#--------------------------------------------------------------------------
#-----------------------Simulate robot motion------------------------------
#--------------------------------------------------------------------------
def simulate_robot(t,xx,xx1,u_cl,xs,N, T, c1, M, S):
    vx0 = xx[0,0]
    vy0 = xx[1,0]
    vth0 = xx[2,0]

    v1 = Vehicle(vx0, vy0, vth0)
    
    vxt = xs[0]
    vyt = xs[1]
    vtht = xs[2]
    
    vt = Vehicle(vxt, vyt, vtht)

    # Figure initialisation
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(15,7))
    plt.xlabel('x coordinate [m]')
    plt.ylabel('y coordinate [m]')
    ax.axis('equal')
    
    # Plot the wall
    ax.plot([0, 4.5], [-0.9, -0.9], '-', color='tab:gray', linewidth=10, label='_nolegend_')
    ax.plot([0, 4.5], [0.9, 0.9], '-', color='tab:gray', linewidth=10)

    # Plot the robot
    plt1, = ax.plot(v1.x, v1.y, "+b", markersize=50.2)
    plt2, = ax.plot(v1.x, v1.y, "ob", markersize=10.2)
    plt3, = ax.plot(v1.gvx, v1.gvy, "-b")

    # Plot the casualty
    plt4, = ax.plot(c1.x, c1.y, "+r", markersize=25.2)
    plt5, = ax.plot(c1.x, c1.y, "or", markersize=5.2)
    plt6, = ax.plot(c1.gcx, c1.gcy, "-b")

    # Plot the target pose
    # Calculate target
    target_pose = TargetVehicle(c1)
    plt7, = ax.plot(target_pose.x, target_pose.y, "+r", markersize=50.2)
    plt8, = ax.plot(target_pose.x, target_pose.y, "or", markersize=10.2)
    plt9, = ax.plot(target_pose.gtx, target_pose.gty, "--r")

    # Plot the heading line
    a,b,c = cal_heading_line(target_pose)
    x0 = 0.0
    x1 = 4.5
    y0 = ((a*x0)+c)/(-b)
    y1 = ((a*x1)+c)/(-b)
    plt10, = ax.plot([x0,x1], [y0,y1], ":b")

    # Plot trajectory
    plt15, = ax.plot(xx[0,:], xx[1,:], ":r")

    # Plot the initial robot pose
    initial_pose = Vehicle(vx0, vy0, vth0)
    plt16, = ax.plot(initial_pose.x, initial_pose.y, "+b", markersize=50.2)
    plt17, = ax.plot(initial_pose.x, initial_pose.y, "ob", markersize=10.2)
    plt18, = ax.plot(initial_pose.gvx, initial_pose.gvy, "--b")

    # Animation
    for k in range(xx.shape[1]):
        v1 = Vehicle(xx[0,k], xx[1,k], xx[2,k])

        plt1.set_data(v1.x, v1.y)
        plt2.set_data(v1.x, v1.y)
        plt3.set_data(v1.gvx, v1.gvy)

        # Plot the state prediction
        if k ==0:
            px = np.array([])
            py = np.array([])
            plt12, = ax.plot(px, py, ".b", markersize=10.2)
            plt13, = ax.plot(px, py, "--b")
        else:
            px = xx1[0,:,k-1]
            py = xx1[1,:,k-1]
            plt12.set_data(px, py)
            plt13.set_data(px, py)

        plt.pause(T)
        plt.show
    plt.pause(1.5)
    plt.axis('off')
    path = "results/"+M+"_"+S+"_sim_trajectory.png"
    fig.savefig(os.path.abspath(path))

    # Plot the Path
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,5), sharex=True)
    ax1.plot(t, xx[0,:], '-')
    ax1.set_ylabel('x [m]')
    ax1.get_yaxis().set_label_coords(-0.085,0.5)
    ax1.set_ylim(0,4)
    ax1.set_xlim(0,20)
    ax1.grid()

    ax2.plot(t, xx[1,:], '-')
    ax2.set_ylabel('y [m]')
    ax2.get_yaxis().set_label_coords(-0.085,0.5)
    ax2.grid()

    ax3.plot(t, xx[2,:], '-')
    ax3.set_ylabel(r"$\theta$ [rad]")
    plt.xlabel('time [s]')
    ax3.get_yaxis().set_label_coords(-0.085,0.5)
    ax3.grid()
    path = "results/"+M+"_"+S+"_state_trajectory.png"
    fig1.savefig(os.path.abspath(path))
    

    # Plot the control
    fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(10,5), sharex=True)
    ax4.step(t, xx[3,:]/0.016, '-')
    ax4.set_ylabel(r"$v$ [m/s]")
    ax4.get_yaxis().set_label_coords(-0.1,0.5)
    ax4.grid()
    ax4.set_xlim(0,20)

    ax5.step(t, xx[4,:]/0.011, '-')
    ax5.set_ylabel(r"$\omega$ [rad/s]")
    ax5.get_yaxis().set_label_coords(-0.1,0.5)
    ax5.grid()
    plt.xlabel('time [s]')
    plt.show()
    path = "results/"+M+"_"+S+"_control_trajectory.png"
    fig2.savefig(os.path.abspath(path))

def main():
    print("start!!")
    print("done!!")

if __name__ == '__main__':
    main()