import numpy as np
from matplotlib import pyplot as plt

class Logger2():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['real', item, level] = []

    def log_data(self, corner_l, corner_r,current):
        for item in corner_l:
                self.log['real', 'corner_left', item].append(corner_l[item])
        for item in corner_r:
                self.log['real', 'corner_right', item].append(corner_r[item])
        for item in current.keys():
            for level in current[item].keys():
                self.log['real', item, level].append(current[item][level])

    def initialize_plot(self, frequency=1):
        self.frequency = frequency
        self.plot_info = [
            {'axis': 0, 'batch': 'real', 'item': 'corner_left', 'level': 'up_right', 'color': 'blue', 'style': 'o'},  
            {'axis': 0, 'batch': 'real', 'item': 'corner_left', 'level': 'up_left', 'color': 'red', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_left', 'level': 'down_left', 'color': 'green', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_left', 'level': 'down_right', 'color': 'purple', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_right', 'level': 'up_left', 'color': 'cyan', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_right', 'level': 'up_right', 'color': 'magenta', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_right', 'level': 'down_left', 'color': 'yellow', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'corner_right', 'level': 'down_right', 'color': 'black', 'style': 'o'},
            {'axis': 0, 'batch': 'real', 'item': 'com', 'level': 'pos', 'color': 'blue', 'style': 'o'}
        ]
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel(' X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('foot Corner')

        self.lines = {}
        for item in self.plot_info:
             key = item['batch'], item['item'], item['level']
             self.lines[key], = self.ax.plot([], [], color=item['color'], linestyle='None', marker=item['style'], markersize=8)
             if key[1] == 'com' :
               self.lines[key], = self.ax.plot([], [], color=item['color'], linestyle='None', marker=item['style'], markersize=3)
   

        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-1, 1)

        plt.ion()
        plt.show()

    def update_plot(self, time):
      if time % self.frequency != 0:
        return

      corner_left_x = []
      corner_left_y = []
      corner_right_x = []
      corner_right_y = []
      com_x = []
      com_y = []

      for item in self.plot_info:
          trajectory_key = item['batch'], item['item'], item['level']

          if item['item'] == 'corner_left':
              data = self.log['real', 'corner_left', item['level']]
              if len(data) > 0:
                  last_point = data[-1]
                  if last_point[2] < 0:
                      corner_left_x.append(last_point[0])
                      corner_left_y.append(last_point[1])

          elif item['item'] == 'corner_right':
              data = self.log['real', 'corner_right', item['level']]
              if len(data) > 0:
                  last_point = data[-1]
                  if last_point[2] <= 0.01:
                      corner_right_x.append(last_point[0])
                      corner_right_y.append(last_point[1])

          elif item['item'] == 'com':  
              data = self.log['real', 'com', item['level']]
              if len(data) > 0:
                  com_x = [point[0] for point in data]  
                  com_y = [point[1] for point in data] 

   
      for item in self.plot_info:
          trajectory_key = item['batch'], item['item'], item['level']
          if item['item'] == 'corner_left':
              self.lines[trajectory_key].set_data(corner_left_x, corner_left_y)
          elif item['item'] == 'corner_right':
              self.lines[trajectory_key].set_data(corner_right_x, corner_right_y)

    
      if com_x and com_y:
          trajectory_key = ('real', 'com', 'pos')  
          self.lines[trajectory_key].set_data(com_x, com_y)  

    
      self.ax.relim()
      self.ax.autoscale_view() 
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()
