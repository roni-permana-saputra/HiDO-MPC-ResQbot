from casadi import *
import numpy as np
import time
import math



class MPC_single_shooting_multi_objective():
    def __init__(self, map, init_vehicle, casualty, obs_list, T, N, v_max, v_min, w_max, w_min):
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
        self.casualty = casualty
        self.vehicle = init_vehicle
        self.obs_list = obs_list
        # self.filtered_obs = self._filter_obstacle()
        self.filtered_obs = obs_list

        self.v_max = v_max 
        self.v_min = v_min
        self.w_max = w_max
        self.w_min = w_min

        self.x_min = map.xmin 
        self.x_max = map.xmax 
        self.y_min = map.ymin 
        self.y_max = map.ymax 

        # Weight
        self.aaa = 97.97446947833316
        self.bbb = 95.65855112916476
        self.ccc = 1.6343820521161234
        self.ddd = 97.29106408885029
        self.eee = 2.494204319362573
        self.fff = 3.8122508089174776

        self._form_model()
        self._form_obj()
        self._form_const()
        self._form_OPT_variables()
        self.form_args()

    def _filter_obstacle(self):
        if self.obs_list !=[]:
            vehicle_node = np.array([self.vehicle.x, self.vehicle.y])
            obs_nodes = np.zeros((len(self.obs_list),2))
            for i in range(len(self.obs_list)):
                obs_nodes[i,0]=self.obs_list[i].x0
                obs_nodes[i,1]=self.obs_list[i].y0
            dist = np.sqrt(np.sum((obs_nodes - vehicle_node)**2, axis=1))
            filtered_obs_idx = np.where(dist < 2.5)
            filtered_obs_idx = filtered_obs_idx[0]
            filtered_obs_idx.astype(int)
            filtered_obs = []
            for idx in filtered_obs_idx:
                filtered_obs.append(self.obs_list[idx])
            return filtered_obs
        return []

    # -------------------------
    # Model Formulation
    # -------------------------
    def _form_model(self):
        # Robot State
        x = SX.sym('x') 
        y = SX.sym('y') 
        theta = SX.sym('theta')
        # v = SX.sym('v')
        # w = SX.sym('w')
        # states = vertcat(x, y, theta, v, w)
        states = vertcat(x, y, theta)
        n_states = states.size(1)

        # Control State
        # a = SX.sym('a') 
        # alpha = SX.sym('alpha')
        # controls = vertcat(a, alpha)
        v = SX.sym('v')
        w = SX.sym('w')
        controls = vertcat(v, w)
        n_controls = controls.size(1)

        # State Transition Function
        rhs = vertcat(v*cos(theta), v*sin(theta), w) # system r.h.s

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
        X[:,0] = P[0:3] # initial state
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

        Q = SX.zeros(3,3); Q[0,0] = 1
        Q[1,1] = 5;  Q[2,2] = 0.1

        R = SX.zeros(2,2)
        R[0,0] = 0.1;  R[1,1] = 0.01 

        # Compute Objective
        self.obj = 0 
        for k in range(self.N):
            st = self.X[:,k]  
            con = self.U[:,k]
            target = self.P[3:6]
            # Calculate Obstacle cost
            gcrx, gcry, gcrr = self.calc_global_four_circles(st[0:3])
            obs_obj = 0
            for j in range(len(gcrx)):
                for obs in self.filtered_obs:
                    dist = np.sqrt(((obs.x0-gcrx[j])**2)+((obs.y0-gcry[j])**2))-(obs.radius+gcrr[j]+0.1)
                    bool_weight = 1/(1+np.exp(60*(dist)))
                    obs_obj =  obs_obj + bool_weight*150#*(1/(dist**2))

            self.obj = (self.obj+self.aaa*self.cal_dist2line_cost(st[0:3], target[0:3])
                                +self.aaa*self.cal_dist2target_cost(st[0:3], target[0:3])
                                +self.aaa*self.cal_dist2endline_cost(st[0:3], target[0:3])
                                +self.bbb*self.cal_dist2target_cost(st[0:3], target[0:3])
                                +25*self.ccc*self.cal_angleDiff_cost(st[0:3], target[0:3])
                                +obs_obj
                                +self.fff*mtimes(mtimes(transpose(con),R),con))
        
    def cal_angleDiff_cost(self, current_pose, target_pose):
        # angleDiff = (numpy.arccos(np.cos(current_pose[2]-target_pose[2])))**2
        angleDiff = (current_angle-target_angle)**2
        return angleDiff

    def cal_dist2target_cost(self, current_pose, target_pose):
        dist2target = ((current_pose[0]-target_pose[0])**2)+((current_pose[1]-target_pose[1])**2)
        return dist2target

    def cal_dist2line_cost(self, current_pose, target_pose):
        a,b,c = self.cal_heading_line(target_pose)
        dist2line = (((a*current_pose[0])+(b*current_pose[1])+c)**2)
        return dist2line

    def cal_dist2endline_cost(self, current_pose, target_pose):
        end_line = self.cal_end_line(target_pose)
        dist2endline = ((current_pose[0]-end_line[0])**2)+((current_pose[1]-end_line[1])**2)
        return dist2endline 

    # -------------------------------
    # Constraints Formulation
    # ------------------------------
    # Stacking all constraint variable elements
    def _form_const(self):
        self.g = []

        # 1. Casualty
        for k in range(self.N+1):
            self._calc_global_four_circles(self.X[0:3,k])
            for i in range(len(self.casualty.gccrx)):
                for j in range(len(self.gcrx)):
                    dist = np.sqrt(((self.casualty.gccrx[i]-self.gcrx[j])**2)+((self.casualty.gccry[i]-self.gcry[j])**2))-(self.casualty.gccrr[i]+self.gcrr[j]+0.025)#0.45
                    self.g = vertcat(self.g, dist)
        self.n_casualty_constraints = self.g.shape[0]

        # 2. Map constrain  
        for k in range(self.N+1):
            self._calc_global_four_circles(self.X[0:3,k])
            for x in self.gcrx:
                self.g = vertcat(self.g, x)
        self.n_map_x_constraints = self.g.shape[0]

        for k in range(self.N+1):
            self._calc_global_four_circles(self.X[0:3,k])
            for y in self.gcry:
                self.g = vertcat(self.g, y)
        self.n_map_y_constraints = self.g.shape[0]

        # 3. Obstacles constrain  
        if self.obs_list !=[]:
            for k in range(self.N+1):
                self._calc_global_four_circles(self.X[0:3,k])
                for j in range(len(self.gcrx)):
                    for obs in self.filtered_obs:
                        dist = np.sqrt(((obs.x0-self.gcrx[j])**2)+((obs.y0-self.gcry[j])**2))-(obs.radius+self.gcrr[j]+0.01)
                        self.g = vertcat(self.g, dist)
            self.n_obstacles_to_circles_distances_constraints = self.g.shape[0]
            print self.n_obstacles_to_circles_distances_constraints
       
    def _calc_global_four_circles(self, st):
        cr_x = [-0.1, 0.3, 0.7, 1.2]
        cr_y = [0.0, 0.0, 0.0, 0.0]
        cr_r = [0.4, 0.4, 0.4, 0.4]
        self.gcrx = [(ix * np.cos(st[2,0]) + iy * np.sin(st[2,0])) +
              st[0,0] for (ix, iy) in zip(cr_x, cr_y)]
        self.gcry = [(ix * np.sin(st[2,0]) - iy * np.cos(st[2,0])) +
              st[1,0] for (ix, iy) in zip(cr_x, cr_y)]
        self.gcrr = cr_r

    def calc_global_four_circles(self, st):
        cr_x = [-0.1, 0.3, 0.7, 1.2]
        cr_y = [0.0, 0.0, 0.0, 0.0]
        cr_r = [0.4, 0.4, 0.4, 0.4]
        gcrx = [(ix * np.cos(st[2,0]) + iy * np.sin(st[2,0])) +
              st[0,0] for (ix, iy) in zip(cr_x, cr_y)]
        gcry = [(ix * np.sin(st[2,0]) - iy * np.cos(st[2,0])) +
              st[1,0] for (ix, iy) in zip(cr_x, cr_y)]
        gcrr = cr_r
        return gcrx, gcry, gcrr

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

        # Arguments for casualty constraints
        array_lbg[0:self.n_casualty_constraints,:] = 0.0
        array_ubg[0:self.n_casualty_constraints,:] = inf

        # Arguments for map constraints
        array_lbg[self.n_casualty_constraints+1:self.n_map_x_constraints,:] = self.x_min
        array_ubg[self.n_casualty_constraints+1:self.n_map_x_constraints,:] = self.x_max
        array_lbg[self.n_map_x_constraints+1:self.n_map_y_constraints,:] = self.y_min
        array_ubg[self.n_map_x_constraints+1:self.n_map_y_constraints,:] = self.y_max

        # Arguments for obstacle constraints
        if self.obs_list !=[]:
            array_lbg[self.n_map_y_constraints+1:self.n_obstacles_to_circles_distances_constraints,:] = 0.0
            array_ubg[self.n_map_y_constraints+1:self.n_obstacles_to_circles_distances_constraints,:] = inf

        # Combaining and input to the dictionary
        args['lbg'] =  array_lbg    # lower bound of the states x and y
        args['ubg'] =  array_ubg    # upper bound of the states x and y 

        # 2. inequality function for decission variables
        lbx = np.zeros((2*self.N,1))
        lbx[range(0, 2*self.N, 2),0] = self.v_min 
        lbx[range(1, 2*self.N, 2),0] = self.w_min

        ubx = np.zeros((2*self.N,1))
        ubx[range(0, 2*self.N, 2),0] = self.v_max 
        ubx[range(1, 2*self.N, 2),0] = self.w_max

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
        return sol, self.filtered_obs

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
        if abs(self.casualty.yaw) != np.pi/2:
            tan_theta = np.tan(target_pose[2])
            b = -1
            a = tan_theta
            c = -((a*target_pose[0])+(b*target_pose[1]))
            return a, b, c
        else:
            b = 0
            a = 1
            c = target_pose[0]
            return a, b, c
       
    def cal_end_line(self, target_pose):
        distance = 1.75
        if abs(self.casualty.yaw) != np.pi/2:
            sin_theta = np.sin(target_pose[2])
            cos_theta = np.cos(target_pose[2])
            xe = target_pose[0]-(distance*cos_theta)
            ye = target_pose[1]-(distance*sin_theta)
            end_line = np.array([xe,ye])
            return end_line
        else:
            xe = target_pose[0]
            ye = target_pose[1]+distance
            end_line = np.array([xe,ye])
            return end_line 

def main():
    print("start!!")
    print("done!!")

if __name__ == '__main__':
    main()