import numpy as np
from utils import *
from Bus import bus_initial
class LowLevelEnvironment(object):
    def __init__(self,buses,trans_lines,nb_interval):
        self.buses = buses
        self.trans_lines = trans_lines
        self.nb_interval = nb_interval
        self.obs_dim = sum([bus.get_obs_dim() for bus in buses.values()])
        self.act_dim = 0
        self.act_start_index = {}
        for index,bus in buses.items():
            self.act_start_index[index] = self.act_dim
            self.act_dim += bus.get_act_dim()-1
        for index in trans_lines.keys():
            self.act_start_index[index] = self.act_dim
            self.act_dim += 1
        self.load_scale = 0.005
        self.battery_scale = 0.01
        self.reward_scale = 0.001
        self.error_scale = 0.01
        nb_bus = len(buses)
        
        self.penaltyCoef = np.zeros((nb_interval,nb_bus*8))
        penaltyCoef_pos = np.zeros(4)
        penaltyCoef_pos[3] += 1000.0
        penaltyCoef_neg = np.zeros(4)
        penaltyCoef_neg[2] += 1000.0
        self.penaltyCoef_eval = np.hstack((np.tile(penaltyCoef_pos, nb_bus),np.tile(penaltyCoef_neg, nb_bus)))
        
        self.reset()
        self._training = True
    def train(self):
        self._training = True
    def evaluate(self):
        self._training = False
    def reset(self, error = []):
        bus_initial(self.buses,error)
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
    def make_action(self,action,t =None, error = {}):
        action = action.tolist()
        reward = 0
        if t is None:
            t = self.time_interval
        for index,line in self.trans_lines.items():
            flow_action = action[self.act_start_index[index]]
            flow_action = line.rescale(flow_action)
            line.update_transout(flow_action)
        raw_action = []
        effect_action = []
        for index,bus in self.buses.items():
            bus_action = action[self.act_start_index[index]: self.act_start_index[index]+bus.get_act_dim()-1]
            raw,eff,r = bus.step(bus_action,t,error)
            raw_action.extend(raw)
            effect_action.extend(eff)
            reward += r
        
        return np.array(raw_action),np.array(effect_action),reward
    def penalty(self,raw_action,effect_action):
        violence = np.hstack((raw_action-effect_action,effect_action-raw_action))
        if self._training:
            penalty = -self.penaltyCoef[self.time_interval] @ np.maximum(violence,0)
            self.penaltyCoef[self.time_interval] += 0.00001* np.maximum(violence,0)
            self.penaltyCoef[self.time_interval] = np.maximum(self.penaltyCoef[self.time_interval],0)
        else:
            penalty = -self.penaltyCoef_eval @ np.maximum(violence,0)
        return penalty
    def step(self,action,error = {}):
        raw_action, effect_action,reward = self.make_action(action,error = error)
        penalty = self.penalty(raw_action,effect_action)
        reward *= self.reward_scale
        penalty *= self.reward_scale
        if self.time_interval == self.nb_interval -1:
            self.reset(error)
            return self.observations,reward,penalty
        else:
            self.time_interval += 1
            self.observations = []
            for bus in self.buses.values():
                if bus.storage:
                    self.observations += bus.get_battery(scale = self.battery_scale)
                if bus.load:
                    self.observations += bus.get_load(self.time_interval,scale = self.load_scale)
            self.observations = np.array(self.observations)

            return self.observations,reward,penalty