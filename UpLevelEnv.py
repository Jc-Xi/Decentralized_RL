import numpy as np
from utils import *
class UpLevelEnvironment(object):
    def __init__(self,tie_line,nb_interval):
        self.tie_line = tie_line
        self.nb_interval = nb_interval
        self.buses = {'source':self.tie_line.nodeSource, 'end':self.tie_line.nodeEnd}
        self.obs_dim = sum([bus.get_obs_dim() for bus in self.buses.values()])
        self.act_dim = 1
        self.load_scale = 0.005
        self.battery_scale = 0.01
        self.reward_scale = 0.1
        self.error_scale = 0.01
        
        self.reset()
    def reset(self):
        self.observations = []
        for bus in self.buses.values():
            if bus.storage:
                self.observations += bus.get_battery(scale = self.battery_scale)
            if bus.load:
                self.observations += bus.get_load(0,scale = self.load_scale)
        self.observations = np.array(self.observations)
        self.time_interval = 0
    def get_obs(self):
        return self.observations
    def reward(self):
        mc1,mc2 = [bus.get_mc() for bus in self.buses.values()]
        return -self.reward_scale*np.square(mc1 -mc2)
    def apply_action(self,action):
        flow_action = action[0]
        flow_action = self.tie_line.rescale(flow_action)
        self.tie_line.update_transout(flow_action)
    def step(self):
        r = self.reward()
        self.observations = []
        if self.time_interval < self.nb_interval - 1:
            self.time_interval+=1
            for bus in self.buses.values():
                if bus.storage:
                    self.observations += bus.get_battery(scale = self.battery_scale)
                if bus.load:
                    self.observations += bus.get_load(self.time_interval,scale = self.load_scale)
            self.observations = np.array(self.observations)
        
            return self.observations,r
        else:
            self.reset()
            return self.observations,r
