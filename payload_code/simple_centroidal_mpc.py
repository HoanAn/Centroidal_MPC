import numpy as np
import casadi as cs
print(cs.Importer_load_plugin)
class centroidal_mpc:
  def __init__(self, initial, footstep_planner, params, CoM_ref, contact_trj_l, contact_trj_r):
    # parameters
    self.params = params
    self.N = params['N']-90

    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.mass = params['mass']
    print("total mass:")
    print(self.mass)
    self.g = params['g']
    self.initial = initial
    self.footstep_planner = footstep_planner
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1) # piecewise linear sigmoidal function
    self.k1=20
    self.k2=0.5
    mu= 0.5
    d= params['foot_size']/2

    self.A=cs.DM([[ 1, 0, 0, 0, 0, -d],
                      [-1, 0, 0, 0, 0, -d],
                      [0,  1, 0, 0, 0, -d],
                      [0, -1, 0, 0, 0, -d],
                      [0, 0, 0,  1, 0, -mu],
                      [0, 0, 0, -1, 0, -mu],
                      [0, 0, 0, 0,  1, -mu],
                      [0, 0, 0, 0, -1, -mu]])
    
    self.b=cs.GenDM_zeros(8)


    self.contact_trj_l=contact_trj_l
    self.contact_trj_r=contact_trj_r
    #An: Get the CoM_ref data from Daniele --> thanks
    #self.CoM_ref_planner=CoM_ref

    self.pos_com_ref_x= CoM_ref['pos_x']
    self.pos_com_ref_y= CoM_ref['pos_y']
    self.pos_com_ref_z= CoM_ref['pos_z']

    self.vel_com_ref_x= CoM_ref['vel_x']
    self.vel_com_ref_y= CoM_ref['vel_y']
    self.vel_com_ref_z= CoM_ref['vel_z']

    self.acc_com_ref_x= CoM_ref['acc_x']
    self.acc_com_ref_y= CoM_ref['acc_y']
    self.acc_com_ref_z= CoM_ref['acc_z']
    
    #An: Get all the foot step ref from foot step planner over time stamp
    self.pos_contact_ref_l= footstep_planner.contacts_ref['contact_left']
    self.pos_contact_ref_r= footstep_planner.contacts_ref['contact_right']
        
    with open("pos_contact_ref_l", "w") as file:
      file.writelines(" \n".join(map(str, self.pos_contact_ref_l)))

    with open("pos_contact_ref_right", "w") as file:
      file.writelines(" \n".join(map(str, self.pos_contact_ref_r)))
    # optimization problem setup
    self.opt = cs.Opti()
    p_opts = {"expand": True,"print_time":False}
    s_opts = {"max_iter": 100000,"print_level": False,"tol":0.001}
    #Set up a proper optimal solver
    self.opt.solver('ipopt',p_opts,s_opts) #An: Use different solver, refer to C++ code

    #  # optimization problem
    # self.opt = cs.Opti('conic')
    # p_opts = {"expand": True}
    # s_opts = {"max_iter": 1000, "verbose": False}
    # self.opt.solver("osqp", p_opts, s_opts)
    
    
  #An: Create optimization variable: prefix "opti_" denotes as symbolic variable
    # An: Left vel And right vel: Components of control input -> Decide by mpc
    self.opti_vel_contact_l= self.opt.variable(3, self.N)
    self.opti_vel_contact_r= self.opt.variable(3, self.N)
    # An: Left contact force And right contact force: Components of control input -> Decide by mpc
    self.opti_force_contact_l = self.opt.variable(3,self.N)
    self.opti_force_contact_r = self.opt.variable(3,self.N)
    # An: Left contact torque And right contact torque: Components of control input -> Decide by mpc
    self.opti_torque_contact_l = self.opt.variable(3,self.N)
    self.opti_torque_contact_r = self.opt.variable(3,self.N)

    self.U = cs.vertcat(self.opti_force_contact_l,self.opti_force_contact_r,
                        self.opti_torque_contact_l,self.opti_torque_contact_r,
                        self.opti_vel_contact_l,self.opti_vel_contact_r)
    
    self.opti_wrench_l=cs.vertcat(self.opti_torque_contact_l,
                                  self.opti_force_contact_l)
    
    self.opti_wrench_r=cs.vertcat(self.opti_torque_contact_r,
                                  self.opti_force_contact_r)
    
    #Define the states of centroidal dynamic model. +1 elements for including initial states. thetahat plays a role as a disturbance observer
    self.opti_CoM = self.opt.variable(3, self.N + 1)
    self.opti_dCoM = self.opt.variable(3, self.N + 1)
    self.opti_hw = self.opt.variable(3, self.N + 1)
    self.opti_thetahat = self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_l= self.opt.variable(3, self.N + 1)
    self.opti_pos_contact_r= self.opt.variable(3, self.N + 1)
    
    #An: Concatenate them in to the state matrix (3*num_state, self.N+1)
    self.opti_state= cs.vertcat(self.opti_CoM,self.opti_dCoM,self.opti_hw,self.opti_thetahat,
                          self.opti_pos_contact_l,self.opti_pos_contact_r)

  #An: Create optimization params that must be updated from the simulator or pre-planner during simulation time
    #An Initial state at the beginning of the mpc horizion
    self.opti_x0_param = self.opt.parameter(18) # update every step based on the current value obtained by the simulator (column vector)
    
    #An: Reference Com trj that is a C2 function
    self.opti_com_ref = self.opt.parameter(3*3,self.N) #including pos x,y,z, vel x,y,z, acc x,y,z ref, update every step based on the pre-planner
    #An: Reference contact points, taken from the footstep planner (to put into the cost function)
    self.opti_pos_contact_l_ref = self.opt.parameter(3,self.N)
    self.opti_pos_contact_r_ref = self.opt.parameter(3,self.N)
  
    #An: to track the contact status of the left and right foot. 1 means foot is in contact, 0 means foot is in the swing phase
    self.opti_contact_left = self.opt.parameter(1,self.N)
    self.opti_contact_right = self.opt.parameter(1,self.N)

    #An: Setup multiple shooting:
    #An: Dynamic constraints
    self.opt.subject_to(self.opti_state[:,0]==self.opti_x0_param) #An: Initial constraint
    #An: Centroidal Dynamic constraints in all the horizon self.N
    for i in range(self.N):
      self.opt.subject_to(self.opti_state[:,i+1]== self.opti_state[:,i]+
                           self.delta*self.centroidal_dynamic(self.opti_state[:,i],self.opti_com_ref[:,i],self.opti_contact_left[i],self.opti_contact_right[i],self.U[:,i]))
    
  #An: Set up the constraints =.=  
    #An: Formulate the change coordinate, constraint only in the first instance
    z1= self.opti_CoM[:,1]-self.opti_com_ref[0:3,0]
    z2= self.k1*z1+self.opti_dCoM[:,1] -self.opti_com_ref[3:6,0]
    
    #for i in range(self.N):
    i=0
    force_avg = self.opti_force_contact_l[:,i]*self.opti_contact_left[i]+self.opti_force_contact_r[:,i]*self.opti_contact_right[i]
    
    #force_avg=force_avg/(2*self.N)
    force_avg=force_avg/(1)
    print("size of force avg")
    print(force_avg.shape)
    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g
    #An: Adaptive force u_n
    u_n= self.k1*self.k1*z1-(self.k1+self.k2)*z2- gravity -self.opti_thetahat[:,0]+self.opti_com_ref[6:9,0]

    #An: Lyapunov stability constrains
    #for i in range(self.N):
    #i=0
    self.opt.subject_to(-z1.T@(self.k1*z1)-z2.T@(self.k2*z2)+z1.T@z2+z2.T@(force_avg-u_n)<0.0)

    # # An: angular momentum constraint:
    for i in range(self.N):
      self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=7)
    # for i in range(1,self.N-1):
    
      
    #    self.opt.subject_to(self.opti_hw[:,i].T@self.opti_hw[:,i]<=2*self.opti_hw[:,i-1].T@self.opti_hw[:,i-1])

    
    #An: Force in z must always be positive
    for i in range(self.N):
      self.opt.subject_to(self.opti_force_contact_l[2,i]>=0)
      self.opt.subject_to(self.opti_force_contact_r[2,i]>=0)

    # #An: Force in z must always below some bounded value 
    # for i in range(self.N):
    #   self.opt.subject_to(self.opti_force_contact_l[2,i]*self.opti_contact_left[i]<=600)
    #   self.opt.subject_to(self.opti_force_contact_r[2,i]*self.opti_contact_right[i]<=600)  
    for i in range(self.N):
      self.opt.subject_to(self.opti_CoM[2,i]<=0.77)
      
    
    #An: Test friction cone without rotation matrix -> need to add foot rotation matrix
    for i in range(self.N):
      self.opt.subject_to(self.A @ self.opti_wrench_l[:,i]*self.opti_contact_left[i]<= self.b)
      self.opt.subject_to(self.A @ self.opti_wrench_r[:,i]*self.opti_contact_right[i]<= self.b)

      # self.opt.subject_to(self.opti_force_contact_l[0,i]<= mu*self.opti_force_contact_l[2,i])
      # self.opt.subject_to(self.opti_force_contact_l[0,i]>=-mu*self.opti_force_contact_l[2,i])
      # self.opt.subject_to(self.opti_force_contact_l[1,i]<= mu*self.opti_force_contact_l[2,i])
      # self.opt.subject_to(self.opti_force_contact_l[1,i]>=-mu*self.opti_force_contact_l[2,i])
      # self.opt.subject_to(self.opti_force_contact_r[0,i]<= mu*self.opti_force_contact_r[2,i])
      # self.opt.subject_to(self.opti_force_contact_r[0,i]>=-mu*self.opti_force_contact_r[2,i])
      # self.opt.subject_to(self.opti_force_contact_r[1,i]<= mu*self.opti_force_contact_r[2,i])
      # self.opt.subject_to(self.opti_force_contact_r[1,i]>=-mu*self.opti_force_contact_r[2,i])

    #An: additional constraint in the z-torque
    # fx_l=self.opti_force_contact_l[0,:]
    # fy_l=self.opti_force_contact_l[1,:]
    # fz_l=self.opti_force_contact_l[2,:]
    # tau_x_l=self.opti_torque_contact_l[0,:]
    # tau_y_l=self.opti_torque_contact_l[1,:]
    # tau_z_l=self.opti_torque_contact_l[2,:]

    # fx_r=self.opti_force_contact_r[0,:]
    # print("size fx_r")
    # print(fx_r.shape)
    # fy_r=self.opti_force_contact_r[1,:]
    # fz_r=self.opti_force_contact_r[2,:]
    # tau_x_r=self.opti_torque_contact_r[0,:]
    # tau_y_r=self.opti_torque_contact_r[1,:]
    # tau_z_r=self.opti_torque_contact_r[2,:]

    # tau_z_min_l= -mu*2*d*fz_l+cs.fabs(d*fx_l-mu*tau_x_l)+cs.fabs(d*fy_l-mu*tau_y_l)
    # tau_z_max_l=  mu*2*d*fz_l-cs.fabs(d*fx_l-mu*tau_x_l)-cs.fabs(d*fy_l-mu*tau_y_l)

    # tau_z_min_r= -mu*2*d*fz_r+cs.fabs(d*fx_r-mu*tau_x_r)+cs.fabs(d*fy_r-mu*tau_y_r)
    # tau_z_max_r=  mu*2*d*fz_r-cs.fabs(d*fx_r-mu*tau_x_r)-cs.fabs(d*fy_r-mu*tau_y_r)

    #self.opt.subject_to(tau_z_l<=tau_z_max_l)
    #self.opt.subject_to(tau_z_l>=tau_z_min_l)

    #self.opt.subject_to(tau_z_r<=tau_z_max_r)
    #self.opt.subject_to(tau_z_r>=tau_z_min_r)
    
    #An: Define the cost function
    # still lack of the components to minimize the deviation of forces at the foot vertices (aka foot corners)
    cost = 1000*cs.sumsqr(self.opti_hw[:,1:]) + \
           1*cs.sumsqr(self.opti_CoM[0,1:]-self.opti_com_ref[0,:])+\
           1*cs.sumsqr(self.opti_CoM[1,1:]-self.opti_com_ref[1,:])+\
           20000*cs.sumsqr(self.opti_CoM[2,1:]-self.opti_com_ref[2,:])+\
           100*cs.sumsqr((self.opti_pos_contact_l[:,1:]-self.opti_pos_contact_l_ref)*self.opti_contact_left[i])+\
           100*cs.sumsqr((self.opti_pos_contact_r[:,1:]-self.opti_pos_contact_r_ref)*self.opti_contact_right[i])
           

    self.opt.minimize(cost)

    #An: initialize the state space to collect the real time state value from the simulator
    self.current_state = np.zeros(3*6)
    #An: CoM_acc as the ff for the inverse dynamic controller
    self.model_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                        'hw' : {'val': np.zeros(3)},
                  'theta_hat': {'val': np.zeros(3)},
          'pos_contact_left' : {'val': np.zeros(3)},
          'pos_contact_right': {'val': np.zeros(3)}}
    
#An: Solve the mpc every time step --> That will be very tough
# Main tasks are updating the current state at the beginning of the horizon and
# let the mpc compute the state in the rest of the horizon
#
  def solve(self, current, t):
    #array = row vector
    self.current_state = np.array([current['com']['pos'][0],       current['com']['pos'][1],       current['com']['pos'][2],
                                   current['com']['vel'][0],       current['com']['vel'][1],       current['com']['vel'][2],
                      current['hw']['val'][0],        current['hw']['val'][1],   current['hw']['val'][2],
                      self.model_state['theta_hat']['val'][0], self.model_state['theta_hat']['val'][1], self.model_state['theta_hat']['val'][2],
                                   current['lfoot']['pos'][3],     current['lfoot']['pos'][4],     current['lfoot']['pos'][5],
                                   current['rfoot']['pos'][3],     current['rfoot']['pos'][4],     current['rfoot']['pos'][5],])
    
    
    #An: Update the initial state contrainst
    print("Left- right Planned foot ref ")
    print(self.contact_trj_l[t][0]['pos'][3:6])
    print(self.contact_trj_r[t][0]['pos'][3:6])

    print("Left -right foot current state:")
    print(self.current_state[12:15])
    print(self.current_state[15:18])
    #print("Right foot current state:")
    print("hw current state:")
    print(self.current_state[6:9])
    
    self.opt.set_value(self.opti_x0_param, self.current_state)

    #An: Extract the status of the contacts
    #An: Update the "real time" contact phase from contact planner list for entire horzion self.N steps, 
    contact_status_l=np.empty((0, 1))
    contact_status_r=np.empty((0, 1))
    for i in range(self.N):
      contact_status = self.footstep_planner.get_phase_at_time(t+i)
      #'ds'
      #print("contact_status:")
      #print(contact_status)
      contact_status_l_i=np.array([[1]])
      contact_status_r_i=np.array([[1]])
      if contact_status == 'ss':
        contact_status_l_i=np.array([[0]])
        contact_status_r_i=np.array([[1]])
        contact_status = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t+i)]['foot_id']
        if contact_status=='lfoot':
          contact_status_l_i=np.array([[1]])
          contact_status_r_i=np.array([[0]])

      contact_status_l=np.vstack((contact_status_l,contact_status_l_i))
      contact_status_r=np.vstack((contact_status_r,contact_status_r_i))
    print("planned contact status left -right")
    print(contact_status_l[0])
    print(contact_status_r[0])
    # print("planned contact status right")
    # print(contact_status_r[0])
    with open("update contact status left in entire horizon", "w") as file:
      file.writelines("\n".join(map(str, contact_status_l)))
    with open("update contact status right in entire horizon", "w") as file:
      file.writelines("\n".join(map(str, contact_status_r)))
    #print("contact_status_left:")
    #print(contact_status_l)
    self.opt.set_value(self.opti_contact_left, contact_status_l)
    self.opt.set_value(self.opti_contact_right, contact_status_r)

    #An: Update CoM_ref value for every step t and in an entire horizon N=100
    idx=1
    pos_com_ref_x= self.pos_com_ref_x[t+idx:t+idx+self.N]
    pos_com_ref_y= self.pos_com_ref_y[t+idx:t+idx+self.N]
    pos_com_ref_z= self.pos_com_ref_z[t+idx:t+idx+self.N]

    vel_com_ref_x= self.vel_com_ref_x[t+idx:t+idx+self.N]
    vel_com_ref_y= self.vel_com_ref_y[t+idx:t+idx+self.N]
    vel_com_ref_z= self.vel_com_ref_z[t+idx:t+idx+self.N]

    acc_com_ref_x= self.acc_com_ref_x[t+idx:t+idx+self.N]
    acc_com_ref_y= self.acc_com_ref_y[t+idx:t+idx+self.N]
    acc_com_ref_z= self.acc_com_ref_z[t+idx:t+idx+self.N]

    com_ref_sample_horizon= np.vstack((pos_com_ref_x,pos_com_ref_y,pos_com_ref_z,
                                      vel_com_ref_x,vel_com_ref_y,vel_com_ref_z,
                                      acc_com_ref_x,acc_com_ref_y,acc_com_ref_z))

    print("Com ref pos:")
    print(com_ref_sample_horizon[0:3,0])
    # print("Com ref pos y:")
    # print(com_ref_sample_horizon[1])
    # print("Com ref pos z:")
    # print(com_ref_sample_horizon[2])
    
    self.opt.set_value(self.opti_com_ref,com_ref_sample_horizon)

    #An: Update pos_contact_ref value for every step t and in an entire horizon N=100
    
    pos_contact_ref_l = self.pos_contact_ref_l[t+0:t+0+self.N].T
    #print(pos_contact_ref_l)
    pos_contact_ref_r = self.pos_contact_ref_r[t+0:t+0+self.N].T

    for i in range(self.N):
      self.opt.set_value(self.opti_pos_contact_l_ref[:,i],pos_contact_ref_l[:,i])
      self.opt.set_value(self.opti_pos_contact_r_ref[:,i],pos_contact_ref_r[:,i])

    # for i in range(self.N):
    #   self.opt.set_value(self.opti_pos_contact_l_ref[:,i],self.contact_trj_l[t+i+1][0]['pos'][3:6])
    #   self.opt.set_value(self.opti_pos_contact_r_ref[:,i],self.contact_trj_r[t+i+1][0]['pos'][3:6])
    # solve optimization problem
    
    # self.opt.set_value(self.zmp_x_mid_param, mc_x)
    # self.opt.set_value(self.zmp_y_mid_param, mc_y)
    # self.opt.set_value(self.zmp_z_mid_param, mc_z)

    sol = self.opt.solve()
    self.x = sol.value(self.opti_state[:,1])
    self.u = sol.value(self.U[:,0])

    self.x_collect=sol.value(self.opti_state)
    print("MPC CoM for next step")
    #for i in range(self.N):
    print(self.x_collect[0:3,1])

    with open("mpc contact result in entire horizon", "w") as file:
      #file.writelines("mpc contact result in entire horizon")
      #for i in range(self.N):
        file.writelines(" \n".join(map(str, self.x_collect[12:15,:])))
        #file.writelines("")

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.opti_state, sol.value(self.opti_state))

    model_force_l= self.u[0:3]
    model_force_r= self.u[3:6]
    model_torque_l=self.u[6:9]
    model_torque_r=self.u[9:12]

    # print("mpc result")
    # # print("theta_hat")
    # # print(self.model_state['theta_hat']['val'][0])
    # # print("lfoot")
    # # print(self.model_state['pos_contact_left']['val'])
    # # print("rfoot")
    # # print(self.model_state['pos_contact_right']['val'])
    # print("com")
    # print(self.model_state['com']['pos'])
    # print("com_vel")
    # print(self.model_state['com']['vel'])

    print("Force_l")
    print(model_force_l)
    print("Force_r")
    print(model_force_r)
    # create output LIP state
    # Change the index to take out the result bcz of different order defined in the dynamics model
    self.model_state['com']['pos'] = np.array([self.x[0], self.x[1], self.x[2]])
    self.model_state['com']['vel'] = np.array([self.x[3], self.x[4], self.x[5]])
    self.model_state['com']['acc'] = (model_force_l*contact_status_l[0]+model_force_r*contact_status_r[0]).T/(self.mass)+np.array([0, 0,- self.g])
    self.model_state['hw']['val'] = np.array([self.x[6], self.x[7], self.x[8]])
    self.model_state['theta_hat']['val'] = np.array([self.x[9], self.x[10], self.x[11]])
    self.model_state['pos_contact_left']['val'] = np.array([self.x[12], self.x[13], self.x[14]])
    self.model_state['pos_contact_right']['val'] = (np.array([self.x[15], self.x[16], self.x[17]]))
    self.model_state['hw']['derivative'] = (np.cross(self.x[12:15]-self.x[0:3],model_force_l)+model_torque_l)*contact_status_l[0] + (np.cross(self.x[15:18]-self.x[0:3],model_force_r)+model_torque_r)*contact_status_r[0]
    # (np.cross(np.array([self.x[12], self.x[13], self.x[14]])-np.array([self.x[0], self.x[1], self.x[2]]),model_force_l)+model_torque_l)*contact_status_l[0]+ np.cross(np.array([self.x[15], self.x[16], self.x[17]])-np.array([self.x[0], self.x[1], self.x[2]]),model_force_r)*contact_status_r[0]

    print("CoM_acc")
    print(self.model_state['com']['acc'])


    # Need to add here the output of mpc for contact pos
    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.model_state, contact
  
  def generate_moving_constraint(self, t):
    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
    time_array = np.array(range(t, t + self.N))
    for j in range(len(self.footstep_planner.plan) - 1):
      fs_start_time = self.footstep_planner.get_start_time(j)
      ds_start_time = fs_start_time + self.footstep_planner.plan[j]['ss_duration']
      fs_end_time = ds_start_time + self.footstep_planner.plan[j]['ds_duration']
      fs_current_pos = self.footstep_planner.plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos = self.footstep_planner.plan[j + 1]['pos']
      mc_x += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y, np.zeros(self.N)
  #An's function: Compute the centroidal dynamic using casadi symbolic variables
  #Input:
  # 1/ state: com, dcom, angularmomentum, thetahat, pos_contact_left, pos_contact_right
  #(here I am writing for contact with one vertex --> develop later for multiple vertex)
  # 2.1/ Pre_plan value (numerical value) for building thetahat: pos_CoM_ref, vel_CoM_ref
  # 2.2/ Pre plan value (numerical value) for model: contact_status (on the ground or not)
  # 3/ control input: contact_force, contact_vel
  #Output: Derivative of the state
  # State derivative formular for multiple shooting  
  def centroidal_dynamic(self, state,CoM_ref,contact_lef, contact_right,input):
    k1=self.k1
    mass = self.mass
    #g = np.array([0, 0,- self.g])
    #print(g)
    #g=g.T
    gravity = cs.GenDM_zeros(3)
    gravity[2]=-self.g
    #Extract states
    com=state[0:3]
    pos_left= state[12:15]
    pos_right= state[15:18]
    #Extract inputs
    force_left=input[0:3]
    force_right=input[3:6]
    torque_left=input[6:9]
    torque_right=input[9:12] 
    vel_left= input[12:15]
    vel_right= input[15:18]
    
    CoM_ref_pos= cs.vertcat(CoM_ref[0],CoM_ref[1],CoM_ref[2])
    #CoM_ref_pos=CoM_ref_pos.T

    CoM_ref_vel= cs.vertcat(CoM_ref[3],CoM_ref[4],CoM_ref[5])
    #CoM_ref_vel=CoM_ref_vel.T
    #size=self.N
    # Centroidal dynamic with disturbance estimator theta hat, contact dynamics
    dcom=state[3:6] #state[3],state[4],state[5]
    ddcom= gravity+(1/mass)*(force_left*contact_lef+force_right*contact_right)
    dhw=  (cs.cross(pos_left-com,force_left)+torque_left)*contact_lef+\
          (cs.cross(pos_right-com,force_right)+torque_right)*contact_right
    v_left= (1-contact_lef)*vel_left
    v_right= (1-contact_right)*vel_right
    dthetahat= k1*(com-CoM_ref_pos)+dcom-CoM_ref_vel

    return cs.vertcat(dcom,ddcom,dhw,dthetahat,v_left,v_right)