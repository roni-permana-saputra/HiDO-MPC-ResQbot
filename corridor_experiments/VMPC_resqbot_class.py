""" 
VMPC Class ResQbot 
author: R Saputra
2021
"""
from casadi import *
import numpy as np
import time


def cal_heading_line(target_pose):
    # line define by ax + by + c = 0
    # tan_theta = -(a/b)
    tan_theta = np.tan(target_pose.yaw)
    b = -1
    a = tan_theta
    c = -((a*target_pose.x)+(b*target_pose.y))
    return a, b, c

def cal_constrain_line(target_pose):
    a,b,c = cal_heading_line(target_pose)
    a1 = b
    b1 = -a
    c1 = -((a1*target_pose.x) + (b1*target_pose.y))
    return np.array([a1, b1, c1])

def cal_dist2line(current_pose, target_pose):
    tan_theta = np.tan(target_pose[2])
    b = -1
    a = tan_theta
    c = -((a*target_pose[0])+(b*target_pose[1]))
    dist2line = (fabs((a*current_pose[0])+(b*current_pose[1])+c))/(np.sqrt((a**2)+(b**2)))
    return dist2line


class MPC_single_shooting_multi_objective():
    def __init__(self, T, N):
        # Parameters
        # -----------------------------------
        # Initialisation 
        #   Init all parameters
        #   for formulation
        #   T       : sampling time
        #   N       : prediction horizon
        #   v_max   : maximum linear speed
        #   v_min   : minimum linear speed
        #   w_max   : maximum angular speed
        #   w_min   : minimum angular speed
        # -----------------------------------
        self.T = T
        self.N = N

        # self.v_max = v_max 
        # self.v_min = v_min
        # self.w_max = w_max
        # self.w_min = w_min
        # self.mpc_type = mpc_type

        # Weight
        self.aaa = 100
        self.bbb = 1

        self._form_model()
        self._form_obj()
        self._form_const()
        self._form_OPT_variables()
        self.form_args()

    # -------------------------
    # Model Formulation
    # -------------------------
    def _form_model(self):
        # Robot State
        x = SX.sym('x') 
        y = SX.sym('y') 
        theta = SX.sym('theta')
        v = SX.sym('v')
        w = SX.sym('w')
        states = vertcat(x, y, theta, v, w)
        n_states = states.size(1)

        # Control State
        a = SX.sym('a') 
        alpha = SX.sym('alpha')
        controls = vertcat(a, alpha)
        n_controls = controls.size(1)

        # State Transition Function
        rhs = vertcat(v*cos(theta), v*sin(theta), w, a, alpha) # system r.h.s

        # nonlinear mapping function f(x,u)
        f = Function('f',[states, controls],[rhs]) 
        self.f = f

        # Decission Variables Matrix
        # This case is single-shooting
        # Decision variables only consists of control sequence 
        U = SX.sym('U',n_controls,self.N)   # Matrix U n_controls by N (symbolic)
        self.U = U 

        # Parameters Vector
        # This vector consists of:
        #   1. the initial and 
        #   2. the reference state of the robot
        # P = SX.sym('P',n_states + n_states)
        # self.P = P
        #   3. the rprevious control
        P = SX.sym('P',n_states + n_states)
        self.P = P

        # State Prediction Matrix
        # A Matrix (n_states,(N+1)) that represents 
        # the states over the optimization problem.
        X = SX.sym('X',n_states,(self.N+1))

        # State Prediction Model
        # Compute State Prediction Recursively 
        # based on State Transition Model
        X[:,0] = P[0:5] # initial state
        for k in range(self.N):
            st = X[:,k]  
            con = U[:,k]
            f_value  = f(st,con)
            st_next  = st + (self.T*f_value)
            X[:,k+1] = st_next
        self.X = X

        # Function to calculate optimat prediction trajectory
        # Given optimal function obtained from the optimisation
        ff=Function('ff',[U,P],[X])
        self.ff = ff

    # -------------------------------
    # Objective Function Formulation
    # -------------------------------
    def _form_obj(self):
        # Classical MPC cost function formulation
        # 
        # Obj = SIGMA(State_Deviation_Objectives + Control_Deviation_Objectives)
        # State_Deviation_Objectives = 
        #       mtimes(mtimes(transpose((st-self.P[3:6])),Q),(st-self.P[3:6])) 
        # Control_Deviation_Objectives =
        #       mtimes(mtimes(transpose(con),R),con)
        # Q = weighing matrices (states)
        # R = weighing matrices (controls)

        Q = SX.zeros(5,5); 
        Q[0,0] = 1;   Q[1,1] = 5;  Q[2,2] = 0.1; 
        Q[3,3] =0.1;  Q[4,4] = 0.01

        R = SX.zeros(2,2)
        R[0,0] = 0.1;  R[1,1] = 0.01 

        # Compute Objective
        self.obj = 0 
        for k in range(self.N):
                st = self.X[:,k]  
                con = self.U[:,k]
                self.obj = self.obj+self.aaa*mtimes(mtimes(transpose((st-self.P[5:10])),Q),(st-self.P[5:10]))+self.bbb*mtimes(mtimes(transpose(con),R),con)
        
    def cal_dist2line_cost(self, current_pose, target_pose):
        a,b,c = self.cal_heading_line(target_pose)
        dist2line = (((a*current_pose[0])+(b*current_pose[1])+c)**2)
        return dist2line

    # -------------------------------
    # Constraints Formulation
    # ------------------------------

    # Stacking all constraint variable elements
    def _form_const(self):
        self.g = []

        # 1. Safety constrain
        #       Constraining distance from the robot into safety line
        for k in range(self.N+1):   
            self.g = vertcat(self.g, self.X[0,k])   
        self.n_safety_constraints = self.g.shape[0]

        # 2. Robot geometrics constrain
        #       Constraining all points in vehicle contours
        #       in y direction
        for k in range(self.N+1):
            self._calc_global_vehicle_contour(self.X[0:3,k])
            for j in range(len(self.gvy)):
                self.g = vertcat(self.g, self.gvy[j])
        self.n_geometric_constraints = self.g.shape[0]

        # 3. Linear speed constrain
        for k in range(self.N+1):   
            self.g = vertcat(self.g, self.X[3,k])   
        self.n_linear_speed_constraints = self.g.shape[0]

        # 3. Angular speed constrain
        for k in range(self.N+1):   
            self.g = vertcat(self.g, self.X[4,k])   
        self.n_angular_speed_constraints = self.g.shape[0]


    def _calc_global_vehicle_contour(self, st):
        v_x = [-0.4, -0.3, 1.4, 1.5, 1.4, -0.3, -0.4]
        v_y = [0.0, -0.3, -0.3, 0.0, 0.3, 0.3, 0.0]
        self.gvx = [(ix * np.cos(st[2,0]) + iy * np.sin(st[2,0])) +
              st[0,0] for (ix, iy) in zip(v_x, v_y)]
        self.gvy = [(ix * np.sin(st[2,0]) - iy * np.cos(st[2,0])) +
              st[1,0] for (ix, iy) in zip(v_x, v_y)]

    # -----------------------------------
    # Formulising Non Linear Programming
    # Optimisation Problem
    # -----------------------------------
    def _form_OPT_variables(self):
        # Formulise decision variable
        OPT_variables = reshape(self.U,2*self.N,1)

        # Formulise nlp problem
        # Elements:
        #   1. Objective function   --->  f
        #   2. Decision Variables   --->  'x': OPT_variables
        #   3. Constraints          --->  'g': g
        #   4. Parameter            --->  'p': P
        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}

        # Set Hyperparameter for Optimiser
        #   We use 'ipopt' optimiser
        #   Check the hyperparameter in:
        opts = {'ipopt': {'max_iter': 100, 'print_level': 0, 'print_frequency_time': 0,
                          'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6} }

        # Formulise optimisation solver
        #   Solver              ---> 'nlpsol'
        #   solver_setting      ---> nlp_prob
        #   Optimiser           ---> 'ipopt'
        #   optimiser_setting   ---> opts
        solver = nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.solver = solver

    # -----------------------------------
    # Formulising Arguments
    # Constraints arguments
    # -----------------------------------
    def form_args(self, lbg=-20.0, ubg=20.0 ):
        # Set Arguments as Dictionary on Python
        #   Elements:
        #       1. 'lbg'  ---> Lower bound from g (constraint variables)
        #       2. 'ubg'  ---> Upper bound from g (constraint variables)
        #       3. 'lbx'  ---> Lower bound from x (decision variables)
        #       4. 'ubx'  ---> Upper bound from x (decision variables) 
        args = dict() 

        # 1. inequality function for constraints variables
        array_lbg = np.zeros(self.g.shape)
        array_ubg = np.zeros(self.g.shape)

        # Arguments for safety constraits
        array_lbg[0:self.n_safety_constraints,:] = 0.5
        array_ubg[0:self.n_safety_constraints,:] = 1.5

        # Arguments for robot geometric constraints
        array_lbg[self.n_safety_constraints+1:self.n_geometric_constraints,:] = -0.9
        array_ubg[self.n_safety_constraints+1:self.n_geometric_constraints,:] = 0.9

        # Arguments for robot linear speed constraints
        array_lbg[self.n_geometric_constraints+1:self.n_linear_speed_constraints,:] = -0.5
        array_ubg[self.n_geometric_constraints+1:self.n_linear_speed_constraints,:] = 0.5

        # Arguments for robot linear speed constraints
        array_lbg[self.n_linear_speed_constraints+1:self.g.shape[0],:] = -0.25
        array_ubg[self.n_linear_speed_constraints+1:self.g.shape[0],:] = 0.25

        # Combaining and input to the dictionary
        args['lbg'] =  array_lbg    # lower bound of the states x and y
        args['ubg'] =  array_ubg    # upper bound of the states x and y 


        # 2. inequality function for decission variables
        lbx = np.zeros((2*self.N,1))
        lbx[range(0, 2*self.N, 2),0] = -1.15 #self.v_min 
        lbx[range(1, 2*self.N, 2),0] = -0.5 #self.w_min

        ubx = np.zeros((2*self.N,1))
        ubx[range(0, 2*self.N, 2),0] = 0.5 #self.v_max 
        ubx[range(1, 2*self.N, 2),0] = 0.5 #self.w_max

        # Combaining and input to the dictionary
        args['lbx'] = lbx     # lower bound of the inputs v and omega
        args['ubx'] = ubx     # upper bound of the inputs v and omega 
        self.args = args 

    # -----------------------------------
    # Solving the NLP 
    # -----------------------------------
    def mpc_solver(self, p, u0):
        mpc_x0 = reshape(transpose(u0),2*self.N,1)  # initial value of the optimization variables    
        sol = self.solver(x0=mpc_x0, 
                     lbx= self.args['lbx'],
                     ubx=self.args['ubx'], 
                     lbg=self.args['lbg'], 
                     ubg=self.args['ubg'], 
                     p=p) 
        return sol

    def move_robot(self, x0, u, t0):
        st = x0
        con = transpose(u)[0,:]
        f_value = self.f(st,con)
        st = st + (self.T*f_value)
        x0 = st 
        u0 = vertcat(u[1:u.shape[0],:] , u[u.shape[0]-1,:])
        t0 = t0 + self.T
        return x0, u0, t0

    # -----------------------
    # Problem for casevac
    # -----------------------
    def cal_heading_line(self, target_pose):
        # line define by ax + by + c = 0
        # tan_theta = -(a/b)
        tan_theta = np.tan(target_pose[2])
        b = -1
        a = tan_theta
        c = -((a*target_pose[0])+(b*target_pose[1]))
        return a, b, c

    def cal_constrain_line(self, target_pose):
        a,b,c = cal_heading_line(target_pose)
        a1 = b
        b1 = -a
        c1 = -((a1*target_pose.x) + (b1*target_pose.y))
        return a1, b1, c1

def main():
    print("start!!")
    print("done!!")

if __name__ == '__main__':
    main()