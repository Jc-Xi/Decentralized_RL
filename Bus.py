from utils import *
class Bus(object):
    def __init__(self,index,constraints,nb_interval):
        self.index = index
        self.nb_interval = nb_interval
        self.obj_func = constraints['obj_func']
        self.marginal_price =constraints['marginal_price']
        self.perturbation = constraints['perturbation']
        self.storage = constraints['storage']
        if self.storage:    
            #self.battery_scale = constraints['battery_scale']
            self.se_charging = constraints['se_charging']
            self.se_discharging = constraints['se_discharging']
            self.max_energy = constraints['max_energy']
            self.max_charge = constraints['max_charge']
        self.generator = constraints['generator']
        if self.generator:
            self.max_prod = constraints['max_prod']
        self.load = constraints['load']
        if self.load:
            self.inflexible_load = constraints['inflexible_load']
            self.flexible_load = constraints['flexible_load']
            #self.load_scale = constraints['load_scale']
        self.elastic = constraints['elastic']
        if self.elastic:
            self.max_el = constraints['max_el']
        self.obs_dim = self.storage + self.load*3
        self.act_dim = self.load + self.storage +  self.elastic + self.generator
        self.mc = None
        self.initialize()
        #self.reward_history = deque(maxlen= 100)
        """
        self._training = True
    def train(self):
        self._training = True
    def evaluate(self):
        self._training = False
        """
    def initialize(self,error = None):
        self.defer = 0.0
        self.battery = 0.0
        self.epsilon = 0.0
        self.trans_out = 0.0
        if error is None and self.perturbation:
            self.epsilon = perturbation(0,0)
        elif error is not None:
            self.epsilon = error
    def get_load(self,time, scale= 1):
        return [(self.inflexible_load[time]+self.epsilon)*scale,self.flexible_load[time]*scale,self.defer*scale]
    def get_battery(self,scale = 1):
        return [self.battery*scale]
    def get_obs_dim(self):
        return self.obs_dim
    def get_mc(self):
        if self.mc is not None:
            return self.mc
    def get_act_dim(self):
        return self.act_dim
    def clear_transout(self):
        self.trans_out = 0.0
    def update_transout(self, trans_out):
        self.trans_out += trans_out
    def get_transout(self):
        return self.trans_out
    def step(self,action, t,error = {}):
        raw_action = []
        effect_action = []
        #deferable load
        defer = 0
        if self.load:
            defer = action.pop(0)
            if t == self.nb_interval-1:
                defer = -self.defer
            else:
                lb_defer = -self.defer
                ub_defer = self.flexible_load[t]
                defer = rescale_action(defer,lb_defer,ub_defer)
            self.defer += defer
        raw_action.append(defer)
        effect_action.append(defer)
        #charging/discharging
        charging = 0
        e_charging = 0
        if self.storage:
            charging = action.pop(0)
            if t == self.nb_interval -1:
                charging= e_charging = np.maximum(-self.battery*self.se_discharging,-self.max_charge)
            else:
                charging = rescale_action(charging,-self.max_charge,self.max_charge)
                lb_charging = -self.battery*self.se_discharging
                ub_charging = (self.max_energy-self.battery)/self.se_charging
                e_charging = np.clip(charging,lb_charging,ub_charging)
            if e_charging >0:
                self.battery += e_charging*self.se_charging
            else:
                self.battery += e_charging/self.se_discharging
        raw_action.append(charging)
        effect_action.append(e_charging)
        
        #elastic load and production
        el = 0
        e_el = 0
        prod = 0
        e_prod = 0
        reward = 0
        if self.elastic and not self.generator:
            if self.load:
                el = defer - self.trans_out- sum(self.get_load(t)[:-1]) - e_charging
            else:
                el = - self.trans_out - e_charging
            e_el = np.clip(el,0,self.max_el)
            reward = self.obj_func(e_el)
            self.mc = self.marginal_price(e_el)
        elif not self.elastic and self.generator:
            if self.load:
                prod = self.trans_out +  sum(self.get_load(t)[:-1]) +  e_charging -defer
            else:
                prod = self.trans_out +  e_charging 
            e_prod = np.clip(prod,0,self.max_prod)
            reward = self.obj_func(e_prod)
            self.mc = self.marginal_price(e_prod)
        raw_action.extend([el,prod])
        effect_action.extend([e_el,e_prod])
        if self.perturbation:
            if self.index in error:
                self.epsilon = error[self.index]
            else:
                self.epsilon = perturbation(self.epsilon,t+1)
        #self.reward_history.append(reward)
        self.clear_transout()
        return raw_action,effect_action,reward
def bus_initial(Buses,error = {}):
    for bus in Buses.values():
        if bus in error:
            bus.initialize(error[bus])
        else:
            bus.initialize()
def load_bus(constraints,Buses,bus2area,nb_interval = 24):
    for key,val in constraints.items():
        Buses[bus2area[key]][key] = Bus(key,val,nb_interval)
