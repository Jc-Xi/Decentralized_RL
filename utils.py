import numpy as np
def rescale_action(action,lb,ub):
    return action * (ub- lb) / 2.0 +\
            (lb + ub) / 2.0
def perturbation(pre_value, t):
    if t < 9 or t >= 17:
        return np.clip(np.random.normal(0,10),-100,100)
    else:
        return np.clip(pre_value + 0.5*np.random.normal(0,10),-100,100)