import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import centroidal_mpc_vertices 
import centroidal_mpc
import simple_centroidal_mpc
import footstep_planner
import footstep_planner_vertices 
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger
from logger2 import Logger2
from logger3 import Logger3 
import new


debug_folder= "Debug"
class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'world_time_step': world.getTimeStep(),            
            'ss_duration': 70,
            'ds_duration': 30,
            'first_swing': 'lfoot',
            'µ': 0.5,
            'N': 100,
            'dof': self.hrp4.getNumDofs(),
            'mass': self.hrp4.getMass(), #An: Add the mass of the robot as a default param
            'update_contact': 'YES'
        }

        model='full_model'   ##    model could be 'full model', 'original' or 'simple'
                                  ## if use 'full model' then use centroidal_mpc_vertices 
                                  ## if use 'full model' then use centroidal_mpc
                                  ## if use 'full model' then use simple_centroidal_mpc
        #model='original'
        # model='simple'                         
        #momentum = 'torso + base + l_hip'  ## if you want to consire angular momentum only of that joint
        #momentum='torso'
        #momentum='base'                  # choose as you wish (but for some model , some momentum might given enfeaseble solution)
        #momentum='semi'
        momentum='full'
        real_walk = self.params['update_contact']      #  or 'YES' if you want that the self.desired position are the one compute by mpc
        
        Angular_update='NO'   # or 'YES'  if ypu want that the angolar momentum is updated by the  formula h = Iw  

        acc='NO'    #if acc = 0 then self.desired[torso or base] = np.zeros(3)
                    #if acc = YES   then i update the angular velocity derivative by inverting  the formula 
                    ##      dwh = I@dw + np.cros(w,I@w)      and so compute the desired dw
        track='torso'        # or base

        self.preferences=[model,momentum ,real_walk,Angular_update,acc,track]   # AN code , stop at 1384
        
        #self.preferences=[model,'base' ,real_walk,'YES','YES','base']   better till now, stop at after 1997
        

        #### i try some other configuration that work at least for a few step :
        #self.preferences=['original','base' ,'YES','YES','NO','NO']  #stop after 1274
        #self.preferences=['original','torso' ,'YES','YES','0','NO'] stop after 1274

        #self.preferences=['simple','torso' ,'NO','YES','NO','NO'] and  self.preferences=['simple','torso' ,'NO','NO','NO','NO'] stop after 1276



        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        
        print("time_step:")
        print(self.params['world_time_step'])
              
        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')
        self.l_hip_p= hrp4.getBodyNode('L_HIP_P_LINK')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

   

   
        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)
        self.com_ref = copy.deepcopy(self.initial)
        self.com_ref = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

             # initialize footstep planner
        reference = [(0.1, 0., 0)] * 5 + [(0.1, 0., -0.0)] * 10 + [(0.1, 0., 0.)] * 10
        if self.preferences[0]=='full_model' :
         self.footstep_planner = footstep_planner_vertices.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )
        else :
         self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )
        
    
        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )
        
        pre_feet_traj= self.foot_trajectory_generator.generate_feet_trajectories_pre()
        self.pre_left_traj=pre_feet_traj['lfoot']
        self.pre_right_traj=pre_feet_traj['rfoot']

        file_path=os.path.join(debug_folder, "Pos Lfoot pre trj")
        with open(file_path, "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_left_traj[i][0]['pos'][3:6]))+ "\n")
            
        file_path=os.path.join(debug_folder, "Pos Rfoot pre trj")
        with open(file_path, "w") as file:
            for i in range(len(self.pre_left_traj)):
                file.writelines(" ".join(map(str, self.pre_right_traj[i][0]['pos'][3:6]))+ "\n")

      
        self.ref=new.references(self.foot_trajectory_generator,self.footstep_planner)  
        print("ref_length:")
        #print(len(self.ref['pos_x']))
        #self.ref=new.references(self.foot_trajectory_generator,self.footstep_planner,1)  FOR SEE GRAHP
        
             # initialize MPC controller
        if self.preferences[0]== 'full_model' :
         self.contact_ref= self.footstep_planner.position_contacts_ref
        # print("contact Ref")
        # print(self.contact_ref['contact_left'][199])
        # print(self.contact_ref['contact_left'][201])
        # print(self.contact_ref['contact_right'][199])
        # print(self.contact_ref['contact_right'][300])
        #print(self.contact_ref['contact_right'][199].shape[1])       
        print("foot_step_plan")
        for i in range(len(self.footstep_planner.plan)):
            print(self.footstep_planner.plan[i])
            print()
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref
            )
        if self.preferences[0] == 'full_model' :
          self.centroidal_mpc=centroidal_mpc_vertices.centroidal_mpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref,
            self.pre_left_traj,
            self.pre_right_traj
          )
        elif self.preferences[0] == 'original' :
          self.centroidal_mpc=centroidal_mpc.centroidal_mpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref,
            self.pre_left_traj,
            self.pre_right_traj
          )
        elif self.preferences[0] == 'simple' :
          self.centroidal_mpc=simple_centroidal_mpc.centroidal_mpc(
            self.initial, 
            self.footstep_planner, 
            self.params,
            self.ref,
            self.pre_left_traj,
            self.pre_right_traj
          )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        d = np.zeros(9)
        d[7] = - self.params['world_time_step'] * self.params['g']
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1], \
                      self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
        self.kf = filter.KalmanFilter(block_diag(A, A, A), \
                                      block_diag(B, B, B), \
                                      d, \
                                      block_diag(H, H, H), \
                                      block_diag(Q, Q, Q), \
                                      block_diag(R, R, R), \
                                      block_diag(P, P, P), \
                                      x)

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)
        self.logger2 = Logger2(self.initial)
        self.logger2.initialize_plot(frequency=10)
        self.logger3 = Logger3(self.initial)
        self.logger3.initialize_plot(frequency=10)

        
    def customPreStep(self):
        # create current and desired states
        if  self.time >1300 and self.time < 1400:
            force = np.array([.0, 1.5, -0.0])  # Newtons
            self.base.addExtForce(force)
            self.torso.addExtForce(force)
        
        #print(self.current)
        #update kalman filter
        # u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        # self.kf.predict(u)
        # x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
        #                                     self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
        #                                     self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))
        self.current = self.retrieve_state() 
        #print(self.current)
        #update kalman filter
        # u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        # self.kf.predict(u)
        # x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
        #                                     self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
        #                                     self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))
        
        # # update current state using kalman filter output
        # self.current['com']['pos'][0] = x_flt[0]
        # self.current['com']['vel'][0] = x_flt[1]
        # self.current['zmp']['pos'][0] = x_flt[2]
        # self.current['com']['pos'][1] = x_flt[3]
        # self.current['com']['vel'][1] = x_flt[4]
        # self.current['zmp']['pos'][1] = x_flt[5]
        # self.current['com']['pos'][2] = x_flt[6]
        # self.current['com']['vel'][2] = x_flt[7]
        # self.current['zmp']['pos'][2] = x_flt[8]
        # # update current state using kalman filter output
        # self.current['com']['pos'][0] = x_flt[0]
        # self.current['com']['vel'][0] = x_flt[1]
        # self.current['zmp']['pos'][0] = x_flt[2]
        # self.current['com']['pos'][1] = x_flt[3]
        # self.current['com']['vel'][1] = x_flt[4]
        # self.current['zmp']['pos'][1] = x_flt[5]
        # self.current['com']['pos'][2] = x_flt[6]
        # self.current['com']['vel'][2] = x_flt[7]
        # self.current['zmp']['pos'][2] = x_flt[8]

        # get references using mpc SOLVE THE MPC
        #lip_state, contact = self.mpc.solve(self.current, self.time)
        robot_state, contact= self.centroidal_mpc.solve(self.current, self.time)
        #lip_state, contact = self.centroidal_mpc.solve(self.current, self.time)
        
        self.desired['com']['pos'] = robot_state['com']['pos']
        self.desired['com']['vel'] = robot_state['com']['vel']
        self.desired['com']['acc'] = robot_state['com']['acc']
        self.desired['hw']['val'] = robot_state['hw']['val']
        
        com_ref=np.zeros(3)
        com_ref[0] = self.ref['pos_x'][self.time]
        com_ref[1] = self.ref['pos_y'][self.time]
        com_ref[2] = self.ref['pos_z'][self.time]

        self.com_ref['com']['pos'] = com_ref
        # self.com_ref['com']['pos'][0] = self.ref['pos_x'][self.time]
        # self.com_ref['com']['pos'][0] = self.ref['pos_x'][self.time]
        # self.com_ref['com']['pos'][0] = self.ref['pos_x'][self.time]

        
        #self.desired['zmp']['vel'] = lip_state['zmp']['vel']

        # self.rb_desired['com']['pos'] = robot_state['com']['pos']
        # self.rb_desired['com']['vel'] = robot_state['com']['vel']
        # self.rb_desired['com']['acc'] = lip_state['com']['acc']
        # self.rb_desired['zmp']['pos'] = lip_state['zmp']['pos']
        # self.rb_desired['zmp']['vel'] = lip_state['zmp']['vel']

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
            #for key in ['vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]
            print(f'Pos_contact_desired {foot}')
            print(self.desired[foot]['pos'][3:6])

        print("Next contact pos in contact list:")
        print(self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(self.time)+1]['pos'])

       
        # file_path=os.path.join(debug_folder, "MPC_pose_contact_ref")        
        # with open(file_path, "w") as file:
        #     for i in range(24):
        #         file.write("\n".join(map(str, self.footstep_planner.plan[i]['pos'].T)) + "\n")
        #         file.write("end"+ "\n")
         
        # print("left foot position trj:")
        # print(self.desired['lfoot']['pos'][3:6])
        
        # print("left foot position trj:")
        # print(self.desired['lfoot']['pos'][3:6])

        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.


        # if self.preferences[2] == 'YES'  and self.preferences[0] != 'simple' :
        #         self.desired['lfoot']['pos'][3:6] = robot_state['pos_contact_left']['val'] 
        #         self.desired['rfoot']['pos'][3:6] = robot_state['pos_contact_right']['val']
        #         print("Real walk enable") 
        #         if self.preferences[0] == 'full_model' :
        #            self.desired['lfoot']['pos'][0:3] = robot_state['ang_contact_left']['val'] 
        #            self.desired['rfoot']['pos'][0:3] = robot_state['ang_contact_right']['val']

        inertia=self.current['inertia']['value'] 

        if self.preferences[3] == "YES"  and self.preferences[5] != 'base ':
            self.desired['torso']['vel']=np.linalg.inv(inertia)@robot_state['hw']['val']
        if self.preferences[3] == "YES"  and self.preferences[5] == 'base ':
            self.desired['base']['vel']=np.linalg.inv(inertia)@robot_state['hw']['val']
        if self.preferences[4] == 0  and self.preferences[5] != 'base ':
            self.desired['torso']['acc']=np.zeros(3)
        elif self.preferences[4] == 0  and self.preferences[5] == 'base ':
          self.desired['base']['acc']= np.zeros(3)
        if self.preferences[4] == 'YES'  and self.preferences[5] != 'base ':
            self.desired['torso']['acc']=np.linalg.inv(inertia)@ (robot_state['hw']['dot']-np.cross(self.desired['torso']['vel'],inertia@ self.desired['torso']['vel']))
        if self.preferences[4] == 'YES'  and self.preferences[5] == 'base ':
            self.desired['torso']['acc']=np.linalg.inv(inertia)@ (robot_state['hw']['dot']-np.cross(self.desired['base']['vel'],inertia@ self.desired['base']['vel']))
                    
    
        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])
        
        self.corner_left=self.current['corner_left']
        self.corner_right=self.current['corner_right']
        # log and plot
        self.logger.log_data( self.desired,self.com_ref)
        self.logger.update_plot(self.time)
        self.logger2.log_data(self.corner_left,self.corner_right,self.current)
        self.logger2.update_plot(self.time)
        self.logger3.log_data(self.desired,self.current)
        self.logger3.update_plot(self.time)
        # print("step index:")
        # print(self.footstep_planner.get_step_index_at_time(self.time))
        # print("step phase:")
        # print(self.footstep_planner.get_phase_at_time(self.time))
        #print(self.footstep_planner.get_start_time(self.footstep_planner.get_step_index_at_time(self.time)))
        self.time += 1# the clock that counts the time
        print(self.time)
  

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))

        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # compute total contact force
        force = np.zeros(3)
        for contact in world.getLastCollisionResult().getContacts():
            force += contact.force
            #print(contact.point)
        #print("endfor")
        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
        for contact in world.getLastCollisionResult().getContacts():
            if contact.force[2] <= 0.1: continue
            zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
            zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

        if force[2] <= 0.1: # threshold for when we lose contact
            zmp = np.array([0., 0., 0.]) # FIXME: this should return previous measurement
        else:
            # sometimes we get contact points that dont make sense, so we clip the ZMP close to the robot
            midpoint = (l_foot_position + l_foot_position) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
            zmp[2] = np.clip(zmp[2], midpoint[2] - 0.3, midpoint[2] + 0.3)
        # print("Torso angl m")
        # print(self.torso.getAngularMomentum(com_position))
        # print("Base angl m")
        # print(self.base.getAngularMomentum(com_position))
        # print("Lfoot angl m")
        # print(self.lsole.getAngularMomentum(com_position))
        # print("Rfoot angl m")
        # print(self.rsole.getAngularMomentum(com_position))
        # print("L_hip_p angl m")
        # print(self.l_hip_p.getAngularMomentum(com_position))


        # Get angular momentum

        if self.preferences[1] == 'torso + base + l_hip' :
         angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)+\
                                    self.base.getAngularMomentum(com_position)*1+\
                                    self.lsole.getAngularMomentum(com_position)*0+\
                                    self.rsole.getAngularMomentum(com_position)*0+\
                                    self.l_hip_p.getAngularMomentum(com_position)
        if self.preferences[1] == 'torso' :
            angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)
        if self.preferences[1] == 'base' :
            angular_momentum_at_com=self.base.getAngularMomentum(com_position)
        if self.preferences[1]=='semi' :
                  angular_momentum_at_com=    self.torso.getAngularMomentum(com_position)+\
                                    self.base.getAngularMomentum(com_position)+\
                                    self.lsole.getAngularMomentum(com_position)+\
                                    self.rsole.getAngularMomentum(com_position)+\
                                    self.l_hip_p.getAngularMomentum(com_position)
        if self.preferences[1] == 'full' :
            angular_momentum_at_com=np.zeros(3)
            for body in hrp4.getBodyNodes():
                w_R_link_i=body.getWorldTransform().rotation()
                #angular_momentum_at_com+=w_R_link_i@body.getAngularMomentum(-com_position+body.getCOM())
                angular_momentum_at_com+=w_R_link_i@body.getAngularMomentum(w_R_link_i.T@(com_position-body.getWorldTransform().translation()))
        
         
        
        c_l1,c_l2,c_l3,c_l4=new.compoute_corner(l_foot_position,l_foot_orientation)
        c_r1,c_r2,c_r3,c_r4=new.compoute_corner(r_foot_position,r_foot_orientation)

        if self.preferences[5] != 'base':
         w_R_com=self.hrp4.getBodyNode('torso').getWorldTransform().rotation()
        else : w_R_com=self.hrp4.getBodyNode('body').getWorldTransform().rotation()
        ###############
        inertia_at_com = np.zeros((3, 3))  #CoM_I_CoM
        for body in self.hrp4.getBodyNodes():
            mass_i = body.getMass()
            if mass_i == 0:
                continue
                
            c_i = body.getCOM(withRespectTo=dart.dynamics.Frame.World())#Position of the CoM of link_i
            # distance = vector from Com_pos to c_i
             
            inertia_i_localCoM = body.getInertia().getMoment()  #inertia matrix wrt local CoM frame
            d_fr_i_CoMi =   body.getLocalCOM()
            delta_i = c_i - com_position - d_fr_i_CoMi  #distance from CoM to link frame i
            skew_dfri = np.array([
                [0              ,-d_fr_i_CoMi[2] ,d_fr_i_CoMi[1]],
                [d_fr_i_CoMi[2] , 0              ,-d_fr_i_CoMi[0]],
                [-d_fr_i_CoMi[1], d_fr_i_CoMi[0]        ,0]
            ])
            # inertia_i_local = np.zeros((6, 6))
            inertia_i_local = inertia_i_localCoM + mass_i * skew_dfri @ skew_dfri.T
            # inertia_i_local[:3, 3:] = mass_i * skew_di
            # inertia_i_local[3:, :3] = mass_i * skew_di.T
            # inertia_i_local[3:, 3:] = mass_i * np.identity(3)

            w_R_i = body.getWorldTransform().rotation() #{w}^R_{i} --> Rot mat. describing orient. of link_i frame wrt World frame

            CoM_R_i=w_R_com.T @ w_R_i            # {CoM}^R_{i}={w}^R_{CoM}.T @ {w}^R_{i}

            # Generalized Huygen-steiner th says:
                # [ CoM_I_i=com_R_i * i_I_i * com_R_i^T ] + mass *[ ||dist||^2*Id-dist @ dist.T ]
            
                    #Note CoM_I_i=com_R_i * i_I_i * com_R_i^T=CoM_I_CoM   ok

            inertia_i_wrt_CoM = CoM_R_i @ inertia_i_local @ CoM_R_i.T   # CoM_I_i=com_R_i*i_I_i*com_R_i^T
            inertia_translated = inertia_i_wrt_CoM + mass_i * (
                np.dot(delta_i, delta_i) * np.identity(3) - np.outer(delta_i, delta_i) # + mass*||dist||^2*Id-dist @ dist.T 
             )

            inertia_at_com += inertia_translated
        #print(f'inertia_at_com:\n{inertia_at_com}')
           
        
        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)},
            'hw'   : {'val': angular_momentum_at_com},
            'theta_hat':{'val':np.zeros(3)},
            'corner_left':{'up_left': c_l1, 'up_right': c_l2,'down_left':c_l3,"down_right":c_l4},
            'corner_right':{'up_left': c_r1, 'up_right': c_r2,'down_left':c_r3,"down_right":c_r4},
            'inertia':{'value': inertia_at_com}    
        }

if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    #stair = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "stairs.urdf"))
    box = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "box.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.addSkeleton(box)
    #world.addSkeleton(stair)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(100) # speed up the visualization by 10x
    node.setTargetRealTimeFactor(100) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
