import numpy as np
from utils import *

        # unicyle_reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10   #[dot{x},dot{y},dot{theta}]
        # self.footstep_planner = footstep_planner.FootstepPlanner(
        #    referunicyle_referenceence,
        #    self.initial['lfoot']['pos'],#pose of the lfoot
        #    self.initial['rfoot']['pos'],#pose of the rfoot
        #    self.params
        #    )

        #'lfoot': {'pos': left_foot_pose,        #left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        #  'vel': l_foot_spatial_velocity,
        #  'acc': np.zeros(6)},

class FootstepPlanner:
    def __init__(self, vref, initial_lfoot, initial_rfoot, params):
        
        debug=0
        ############################ debug

        default_ss_duration = params['ss_duration']  #70
        default_ds_duration = params['ds_duration']  #30
        print(f'default_ss_duration: {default_ss_duration}')
        print(f'default_ds_duration: {default_ds_duration}')

        unicycle_pos   = (initial_lfoot[3:5] + initial_rfoot[3:5]) / 2.      
        unicycle_theta = (initial_lfoot[2]   + initial_rfoot[2]  ) / 2.   #angle with respect the z-axis
        support_foot   = params['first_swing']      #rfoot
        self.plan = []     #dictionarie of information for each foot step
        
        #foot_step sequence  creation
        for j in range(len(vref)):   #now 25 velocity commands
            # set step duration
            ss_duration = default_ss_duration
            ds_duration = default_ds_duration

            # exception for first step in which we want a loong double support phase
            if j == 0:
                ss_duration = 0
                ds_duration = (default_ss_duration + default_ds_duration) * 2

            # exception for last step
            # to be added

            #the ref(velocity commands) changes each 100u, and we emulate the trajectory of a
            #virtual unicycle over the time  u \in [0,len(vref)*100]=[0,2500]   u=world_time_step*t
            for i in range(ss_duration + ds_duration):
                if j > 1:
                    unicycle_theta += vref[j][2] * params['world_time_step']                #vref[j][2]=dot{theta}
                    R = np.array([[np.cos(unicycle_theta), - np.sin(unicycle_theta)],
                                  [np.sin(unicycle_theta),   np.cos(unicycle_theta)]])
                    unicycle_pos += R @ vref[j][:2] * params['world_time_step']             #vref[:2]=[dot{x} dot{y}]

            # compute step position, considering that the foot are displaced from the unicycle movment that is taken as the mean line between the two foots
            displacement = 0.1 if support_foot == 'lfoot' else - 0.1
            displ_x = - np.sin(unicycle_theta) * displacement
            displ_y =   np.cos(unicycle_theta) * displacement
            pos = np.array((                            #Position of the foot on the ground
                unicycle_pos[0] + displ_x, 
                unicycle_pos[1] + displ_y,
                0.))
            ang = np.array((0., 0., unicycle_theta))    #orientation of the foot on the ground

            # add step to plan
            self.plan.append({
                'pos'        : pos,
                'ang'        : ang,
                'ss_duration': ss_duration,
                'ds_duration': ds_duration,
                'foot_id'    : support_foot
                })
            
            # switch support foot
            support_foot = 'rfoot' if support_foot == 'lfoot' else 'lfoot'
            #self.plan is the list of points at each control time. We have len(vref)=len(self.plan) so 25 points 

        #Now Generate ref contact position for every instance of time getting 2500 items both for left and rights foot
        self.position_contacts_ref=self.gen_pos_contacts_ref_at_time(params)  
        

        if debug:
            print(f'\n\t*****Footstep_planner_debug*****')
            print(f'self.plan:{self.plan}')
            print(f'len(self.plan):{len(self.plan)},{range(len(self.plan))}')
            print('--------------------------------------------------')  
            print(f'self.position_contacts_ref:{self.position_contacts_ref}')
            print(f'len(self.position_contacts_ref):[{len(self.position_contacts_ref["contact_left"])};{len(self.position_contacts_ref["contact_right"])}]')
            print(f'\t *****End tFootstep_planner_debug*****\n')

    #An: get the number of step           
    def get_step_index_at_time(self, time):
        t = 0
        for i in range(len(self.plan)):
            t += self.plan[i]['ss_duration'] + self.plan[i]['ds_duration']
            if t > time: return i
        return None
    
    #An: get the start time of the step i
    def get_start_time(self, step_index):
        t = 0
        for i in range(step_index):
            t += self.plan[i]['ss_duration'] + self.plan[i]['ds_duration']
        return t

    def get_phase_at_time(self, time):
        step_index = self.get_step_index_at_time(time)
        start_time = self.get_start_time(step_index)
        time_in_step = time - start_time
        if time_in_step < self.plan[step_index]['ss_duration']:
            return 'ss'
        else:
            return 'ds'
    

    def gen_pos_contacts_ref_at_time(self, params):
        first_swing = params['first_swing']
        time_step = params['world_time_step']
        sim_time = int((len(self.plan)) / time_step)  # 2500 simulation steps
        print(f'Sim time: {sim_time}')
        pose_left = []
        pose_right = []

        for i in range(sim_time):
            index = self.get_step_index_at_time(i)
            #print(f'Contact index: {index}')
            if index < 2:
                if first_swing == 'lfoot':
                    pos_left_i = self.plan[2 * index]['pos']
                    ang_left_i = self.plan[2 * index]['ang']
                    pos_right_i = self.plan[1]['pos']
                    ang_right_i = self.plan[1]['ang']
                else:
                    pos_left_i = self.plan[1]['pos']
                    ang_left_i = self.plan[1]['ang']
                    pos_right_i = self.plan[2 * index]['pos']
                    ang_right_i = self.plan[2 * index]['ang']
            else:
                if first_swing == 'lfoot':
                    pos_left_i = self.plan[index + (index % 2)]['pos']
                    ang_left_i = self.plan[index + (index % 2)]['ang']
                    pos_right_i = self.plan[index + (index - 1) % 2]['pos']
                    ang_right_i = self.plan[index + (index - 1) % 2]['ang']
                else:
                    pos_left_i = self.plan[index + (index - 1) % 2]['pos']
                    ang_left_i = self.plan[index + (index - 1) % 2]['ang']
                    pos_right_i = self.plan[index + (index % 2)]['pos']
                    ang_right_i = self.plan[index + (index % 2)]['ang']

            # Concatenate [ang, pos] to form full pose
            pose_left.append(np.hstack((ang_left_i, pos_left_i)))
            pose_right.append(np.hstack((ang_right_i, pos_right_i)))

        return {
            'contact_left': np.array(pose_left),   # [ang, pos] for left foot
            'contact_right': np.array(pose_right)  # [ang, pos] for right foot
        }
        #:return: Dictionary with foot poses (orientation + position) over time
        #{'contact_left': array([[ang->rotation vector, Position], [ang, 1.03109240e-17,  1.00000000e-01,  0.00000000e+00], position of the left foot
        #'contact_right': array([[ang->rotation vector, Position], [ang, 1.03109240e-17, -1.00000000e-01,  0.00000000e+00], position of the right foot

        

        #This code sets two important things:
        # self.plan
        # is the Dictionrie of points at each control time. We have len(vref)=len(self.plan) so 25 points 


        # self.position_contacts_ref
        # Dictionarie containing all the position for every instance of time getting 2500 items both for left and rights foot
        