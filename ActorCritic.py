import numpy as np
class Actor:
  def __init__(self, nb_feature,nb_action):
    self.nb_feature = nb_feature
    self.nb_action = nb_action
    self.weight = np.random.normal(scale = 0.001,size = (nb_feature,nb_action))
    self.sigma = 0.001
  def forward(self,feature):
    #mu = np.clip(feature @ self.weight,a_min = -1.0, a_max = 1.0)
    mu = np.tanh(feature @ self.weight)
    return mu
  def __call__(self,feature):
    return self.forward(feature)
  def select_action(self,feature):
    mu= self.forward(feature)
    action = np.random.normal(loc = 0.0, scale = 1.0,size = (self.nb_action,))
    action = np.clip(mu + action * self.sigma,-1.0,1.0)
    gradient_mu = -(1-mu**2)* 0.5*(mu-action)/self.sigma * np.tile(feature,(self.nb_action,1)).T
    return action,gradient_mu
  def update_weight(self,gradient_mu,step_size):
    self.weight += step_size * gradient_mu
    try:
        self.weight/= np.maximum(1,np.linalg.norm(self.weight,2)/1e3)
    except np.linalg.LinAlgError:
        print(self.weight)
  def update_sigma(self,new_sigma):
    self.sigma = new_sigma
  def decay_sigma(self,decay_rate):
    self.sigma *= decay_rate
class Critic:
  def __init__(self, nb_feature):
    self.weight = np.random.normal(size = nb_feature)
  def __call__(self,feature):
    return feature @ self.weight
  def update_weight(self,update_volume,step_size):
    self.weight += update_volume * step_size
class ActorCritic(object):
  def __init__(self,nb_interval,nb_feature,nb_action):
    #self.constraints = constraints
    self.nb_interval = nb_interval
    self.nb_state = nb_feature
    self.nb_action = nb_action

    self.actor_dict = dict()
    self.critic_dict = dict()
    for i in range(self.nb_interval):
      self.actor_dict[i] = Actor(nb_feature,nb_action)
      self.critic_dict[i] = Critic(nb_feature)
  def select_action(self,s_t,time_interval):
    action,grad_mu = self.actor_dict[time_interval].select_action(s_t)
    return action,grad_mu
  def update_policy(self,state,grad_mu,reward,next_state,time_interval,lr_actor,lr_critic):
    value = self.critic_dict[time_interval](state)
    if time_interval == self.nb_interval -1 :
      target = reward
    else:
      target = self.critic_dict[time_interval+1](next_state) + reward
    advantage = target- value
    self.actor_dict[time_interval].update_weight(grad_mu*advantage,lr_actor)
    self.critic_dict[time_interval].update_weight(advantage*state,lr_critic)
  def update_sigma(self,new_sigma):
    for k in range(self.nb_interval):
      self.actor_dict[k].update_sigma(new_sigma)

  def decay_sigma(self,decay_rate):
    if self.actor_dict[0].sigma > 0.001:
      for k in range(self.nb_interval):
        self.actor_dict[k].decay_sigma(decay_rate)