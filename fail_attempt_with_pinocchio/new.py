import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import casadi as cs
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def compute_knot(foot_tra,planner,first_knot):
        knot_x=[]
        knot_y=[ ]
        sequence_x=[]#[0]
        sequence_y=[]

        ss_duration = planner.plan[2]['ss_duration']
        ds_duration = planner.plan[2]['ds_duration']

        knot_x.append(first_knot[0])
        knot_y.append((first_knot[1]))
        #knot_x.append((foot_tra.generate_feet_trajectories_at_time(0)['lfoot']['pos'][3]+foot_tra.generate_feet_trajectories_at_time(0)['rfoot']['pos'][3])/2)
        #knot_y.append((foot_tra.generate_feet_trajectories_at_time(0)['lfoot']['pos'][3]+foot_tra.generate_feet_trajectories_at_time(0)['rfoot']['pos'][3])/2)
        knot_x.append((foot_tra.generate_feet_trajectories_at_time(0)['lfoot']['pos'][3]+foot_tra.generate_feet_trajectories_at_time(0)['rfoot']['pos'][3])/2)
        contact=planner.plan[1]['foot_id']
        knot_y.append(foot_tra.generate_feet_trajectories_at_time(0)[contact]['pos'][4]*0.6)
        #knot_y.append((foot_tra.generate_feet_trajectories_at_time(0)['lfoot']['pos'][4]+foot_tra.generate_feet_trajectories_at_time(0)['rfoot']['pos'][4])/2)
        
        scale=ss_duration+ds_duration
        first_time_knot=int(2*scale)
        print(f'first_time_knot: {first_time_knot}')
        sequence_x.append(first_time_knot)
        sequence_y.append(first_time_knot)
        first_contact_time=first_time_knot+ss_duration+1
        
        for i in range(first_time_knot,(len(planner.plan)-1)*scale):
         if (i-first_contact_time)%scale==0 :
          knot_x.append((foot_tra.generate_feet_trajectories_at_time(i)['lfoot']['pos'][3]+foot_tra.generate_feet_trajectories_at_time(i)['rfoot']['pos'][3])/2)
          sequence_x.append(i)

          # idx_x=planner.get_step_index_at_time(i)
          # contact=planner.plan[idx_x+1]['foot_id']
          # sequence_x.append(i+30)
          idx_y=planner.get_step_index_at_time(i)
          print(i)
          print("step id")
          print(idx_y)
          contact=planner.plan[idx_y+1]['foot_id']
          # knot_y.append(foot_tra.generate_feet_trajectories_at_time(i)[contact]['pos'][4]*0.5)
          # sequence_y.append(i)
          knot_y.append(foot_tra.generate_feet_trajectories_at_time(i)[contact]['pos'][4]*0.6)
          # knot_y.append((foot_tra.generate_feet_trajectories_at_time(i)['lfoot']['pos'][4]+foot_tra.generate_feet_trajectories_at_time(i)['rfoot']['pos'][4])/2)
          # sequence_y.append(i)
          sequence_y.append(i+int(ds_duration))
          
        t=(len(planner.plan)-1)*scale
        knot_x.append((foot_tra.generate_feet_trajectories_at_time(t)['lfoot']['pos'][3]+foot_tra.generate_feet_trajectories_at_time(t)['rfoot']['pos'][3])/2)
        knot_y.append((foot_tra.generate_feet_trajectories_at_time(t)['lfoot']['pos'][4]+foot_tra.generate_feet_trajectories_at_time(t)['rfoot']['pos'][4])/2)
        sequence_x.append(t)
        sequence_y.append(t)
        plot_spline(knot_x)
        plot_spline_y(knot_y)
        return knot_x,knot_y,sequence_x,sequence_y


def references(foot_tra,planner,first_knot,SHOW_PLOT=1):
         knot_x,knot_y,sequence_x,sequence_y=compute_knot(foot_tra,planner,first_knot)
         co_x = quintic_spline(knot_x)  # Get optimal coefficients from solver
         co_x = np.array(co_x.full()) 
         co_y = quintic_spline(knot_y)  # Get optimal coefficients from solver
         co_y = np.array(co_y.full())
      
        #plot_spline(self.knot_x)
        #plot_spline(self.knot_y)
         ref_pos_x=built_the_reference(knot_x,sequence_x,co_x)
        #for i,num in enumerate(self.sequence) :
            #print(ref[num])
            #print(self.knot_x[i])
        #print(ref[2300:2400])
         ref_pos_x = np.concatenate(ref_pos_x).tolist()
         ref_vel_x=built_the_velocity(knot_x,sequence_x,co_x)
         ref_vel_x = np.concatenate(ref_vel_x).tolist()
         ref_acc_x=built_the_acceleration(knot_x,sequence_x,co_x)
         ref_acc_x = np.concatenate(ref_acc_x).tolist()
         

         ref_pos_y=built_the_reference(knot_y,sequence_y,co_y)
         ref_pos_y = np.concatenate(ref_pos_y).tolist()
         ref_vel_y=built_the_velocity(knot_y,sequence_y,co_y)
         ref_vel_y = np.concatenate(ref_vel_y).tolist()
         
         ref_acc_y=built_the_acceleration(knot_y,sequence_y,co_y)
         ref_acc_y = np.concatenate(ref_acc_y).tolist()

         ref_pos_z = np.full(len(ref_pos_x),0.72)
         ref_vel_z = np.zeros(len(ref_pos_x))
         ref_acc_z = np.zeros(len(ref_pos_x))


         if SHOW_PLOT == 0 :
           easy_plot(ref_pos_x,"ref_pos_x")
           easy_plot(ref_vel_x,"ref_velocity_x")
           easy_plot(ref_acc_x,"ref_acc_x")
           easy_plot(ref_pos_y,"ref_plot_y")
           easy_plot(ref_vel_y,"ref_vel_y_y")
           easy_plot(ref_acc_y,"ref_acc_y")
           #easy_plot_2d(ref_pos_x, ref_pos_y)
         ref = {
                  "pos_x": ref_pos_x,
                   "vel_x": ref_vel_x,
                   "acc_x": ref_acc_x,
                    "pos_y": ref_pos_y,
                    "vel_y": ref_vel_y,
                    "acc_y": ref_acc_y,
                    "pos_z": ref_pos_z,
                    "vel_z": ref_vel_z,
                    "acc_z": ref_acc_z

                    }
         return ref




def quintic_spline(x): 
        n=len(x)
        p=cs.MX.sym('p',6*n)
        c=[]


        for i in range (0,n-1):     ### position costrain
            c.append(x[i]-p[6*i])
            c.append(x[i+1]- ( p[6*i]+p[6*i+1]+p[6*i+2]+p[6*i+3]+p[6*i+4]+p[6*i+5]))


        c.append(p[1])       ##velocity initial and final constrain 
        c.append(p[6*(n-1)+1])
        for i in range (0,n-1):
            c.append(p[6*i+1]+2*p[6*i+2]+3*p[6*i+3]+4*p[6*i+4]+5*p[6*i+5]-p[6*(i+1)+1])    ##velocity constrain fort continuity


        c.append(2*p[2] )      ##acc initial and final constrain
        c.append(2*p[6*(n-1)+2])
        for i in range (0,n-1):
            c.append(2*p[6*i+2]+6*p[6*i+3]+12*p[6*i+4]+20*p[6*i+5]-2*p[6*(i+1)+2])       ##acceleratino constrain for continuity


        contrain_eq=cs.vertcat(*c)
        nlp = {'x': p, 'f': 0, 'g': contrain_eq}
        solver = cs.nlpsol('solver', 'ipopt', nlp)
        sol = solver(lbg=0, ubg=0)   
        p_opt = sol['x']
        return p_opt

def quintic_spline_y(x): 
        n=len(x)
        p=cs.MX.sym('p',6*n)
        c=[]


        for i in range (0,n-1):     ### position costrain
            c.append(x[i]-p[6*i])
            c.append(x[i+1]- ( p[6*i]+p[6*i+1]+p[6*i+2]+p[6*i+3]+p[6*i+4]+p[6*i+5]))


        c.append(p[1])       ##velocity initial and final constrain 
        i=0
        c.append(p[6*i+1]+2*p[6*i+2]+3*p[6*i+3]+4*p[6*i+4]+5*p[6*i+5])
        #c.append(p[6*(n-1)+1])
        for i in range (1,n-3):
            c.append(p[6*i+1]+2*p[6*i+2]+3*p[6*i+3]+4*p[6*i+4]+5*p[6*i+5])    ##velocity constrain fort continuity
            c.append(p[6*(i+1)+1])

        c.append(2*p[2] )      ##acc initial and final constrain
        c.append(2*p[6*(n-1)+2])
        for i in range (0,n-1):
            c.append(2*p[6*i+2]+6*p[6*i+3]+12*p[6*i+4]+20*p[6*i+5]-2*p[6*(i+1)+2])       ##acceleratino constrain for continuity


        contrain_eq=cs.vertcat(*c)
        nlp = {'x': p, 'f': 0, 'g': contrain_eq}
        solver = cs.nlpsol('solver', 'ipopt', nlp)
        sol = solver(lbg=0, ubg=0)   
        p_opt = sol['x']
        return p_opt



def plot_spline(knot):  
  p_coeff = quintic_spline(knot)  # Get optimal coefficients from solver
  p_coeff_numpy = np.array(p_coeff.full())  # Convert to NumPy array
#print(p_coeff_numpy )
# Plotting the spline 
  n = len(knot) - 1  # Number of segments
  t_vals = np.linspace(0, 1, 100)  # 100 points per segment
  y_vals = []

  for i in range(n):
     # Extract coefficients for the i-th segment
      a0, a1, a2, a3, a4, a5 = p_coeff_numpy[6*i:6*i+6]  # Coefficients of segment i
      y_segment = a0 + a1*t_vals + a2*t_vals**2 + a3*t_vals**3 + a4*t_vals**4 + a5*t_vals**5
      y_vals.extend(y_segment)  # Add segment values to the list

# Generate x-axis values for the plot
  x_plot = np.linspace(0, n, len(y_vals))

# Plot the quintic spline and control points
  plt.plot(x_plot, y_vals, label="Quintic Spline", linewidth=2)
  plt.scatter(range(n+1), knot, color='red', label="Control Points")  # Known points
  plt.legend()
  plt.xlabel("Interval")
  plt.ylabel("Value")
  plt.title("Quintic Spline")
  plt.grid()
  plt.show()

def plot_spline_y(knot):  
  p_coeff = quintic_spline_y(knot)  # Get optimal coefficients from solver
  p_coeff_numpy = np.array(p_coeff.full())  # Convert to NumPy array
#print(p_coeff_numpy )
# Plotting the spline 
  n = len(knot) - 1  # Number of segments
  t_vals = np.linspace(0, 1, 1000)  # 100 points per segment
  y_vals = []

  for i in range(n):
     # Extract coefficients for the i-th segment
      a0, a1, a2, a3, a4, a5 = p_coeff_numpy[6*i:6*i+6]  # Coefficients of segment i
      y_segment = a0 + a1*t_vals + a2*t_vals**2 + a3*t_vals**3 + a4*t_vals**4 + a5*t_vals**5
      y_vals.extend(y_segment)  # Add segment values to the list

# Generate x-axis values for the plot
  x_plot = np.linspace(0, n, len(y_vals))

# Plot the quintic spline and control points
  plt.plot(x_plot, y_vals, label="Quintic Spline", linewidth=2)
  plt.scatter(range(n+1), knot, color='red', label="Control Points")  # Known points
  plt.legend()
  plt.xlabel("Interval")
  plt.ylabel("Value")
  plt.title("Quintic Spline")
  plt.grid()
  plt.show()






def built_the_reference(knot,sequence,p_coeff):
      # p_coeff = quintic_spline(knot)  # Get optimal coefficients from solver
      # p_coeff = np.array(p_coeff.full()) 
      reference=[]
      tick=0
      for i,intervall in enumerate(sequence) :
           for second in range(0,intervall-tick):
                tau=(second)/(intervall-tick)
                a0, a1, a2, a3, a4, a5 = p_coeff[6*i:6*i+6]
                value= a0 + a1*tau + a2*tau**2 + a3*tau**3 + a4*tau**4 + a5*tau**5

                reference.append(value)
           tick=intervall
      return reference


def built_the_velocity(knot,sequence,p_coeff):
      # p_coeff = quintic_spline(knot)  # Get optimal coefficients from solver
      # p_coeff = np.array(p_coeff.full()) 

      reference=[]
      tick=0
      for i,intervall in enumerate(sequence) :
           for second in range(0,intervall-tick):
                tau=(second)/(intervall-tick)
                a0, a1, a2, a3, a4, a5 = p_coeff[6*i:6*i+6]
                value= a1 + 2*a2*tau+ 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4


                reference.append(value)
           tick=intervall
      return reference


def built_the_acceleration(knot,sequence,p_coeff):
      # p_coeff = quintic_spline(knot)  # Get optimal coefficients from solver
      # p_coeff = np.array(p_coeff.full()) 
      # print(sequence)
      # T=np.sum(sequence)
      # print(T)
      # print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
      reference=[]
      tick=0
      for i,intervall in enumerate(sequence) :
           for second in range(0,intervall-tick):
                tau=(second)/(intervall-tick)
                a0, a1, a2, a3, a4, a5 = p_coeff[6*i:6*i+6]
                # value= a2*tau+ 6*a3*tau+ 12*a4*tau**2 + 20*a5*tau**3
                value = (2*a2 + 6*a3*tau + 12*a4*tau**2 + 20*a5*tau**3) / ((intervall - tick) ** 2)


                reference.append(value)
           tick=intervall
      return reference

def easy_plot(y,str):

  x = list(range(len(y))) 
  plt.scatter(x, y, s=0.3, color='b', alpha=0.7, label="value")
  plt.xlabel("Index")  
  plt.ylabel("Value")  
  plt.title(str)  
  plt.grid(True)  
  plt.show()  


def easy_plot_2d(y1, y2, title="Trajectory"):
    plt.figure(figsize=(8, 6))

    plt.scatter(y1, y2, s=0.3,color='b', alpha=0.7, label="")  
    plt.xlabel("x")  
    plt.ylabel("y")  
    plt.title(title)  
    plt.grid(True)  
    plt.legend()  
    plt.show()


def compoute_corner(foot_position, foot_orientation, foot_length=0.27, foot_width=0.15):
    theta = foot_orientation[2] 
    
    half_length = foot_length / 2
    half_width = foot_width / 2
    
    local_corners = np.array([
        [ half_length,  half_width, 0], 
        [-half_length,  half_width, 0],  
        [-half_length, -half_width, 0],
        [ half_length, -half_width, 0]   
    ])
    
  
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [0,               0,               1]
    ])
    

    global_corners = np.dot(local_corners, R.T) + foot_position
    
    return global_corners[0], global_corners[1], global_corners[2], global_corners[3]
