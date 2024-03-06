import numpy as np
#upper_step
def upper_step(poly,agent,env,t):
    obs = env.get_obs()
    obs = poly.fit_transform(obs.reshape(1,-1))[0]
    act = agent.actor_dict[t](obs)
    env.apply_action(act)
#lower_step
def lower_step(poly,area,agent,env,t,boundaries,error = {},load_scale=0.005):
    obs = env.get_obs()
    b1,b2 = boundaries[area]
    obs = np.hstack((obs,env.buses[b1].trans_out*load_scale,env.buses[b2].trans_out*load_scale))
    obs = poly.fit_transform(obs.reshape(1,-1))[0]
    act = agent.actor_dict[t](obs)
    _,r,p = env.step(act,error)
    return r,p
def eval(error_sce,poly,AREA_AGENT,LINE_AGENT,LENV,UENV,boundaries,nb = 10,nb_interval = 24):
        totalReward = 0.0
        totalPenalty = 0.0
        for i in range(nb):
            epsilon_sce = error_sce[i]
            for env in LENV.values():
                 env.reset(epsilon_sce[0])
            for t in range(nb_interval):
                for line in LINE_AGENT.keys():
                    upper_step(poly,LINE_AGENT[line],UENV[line],t)
                for area in AREA_AGENT.keys():
                    if t < nb_interval -1:
                        reward,penalty = lower_step(poly,area,AREA_AGENT[area],LENV[area],t,boundaries,error=epsilon_sce[t+1])
                    else:
                        reward,penalty = lower_step(poly,area,AREA_AGENT[area],LENV[area],t,boundaries)
                    totalReward += reward +penalty
                    totalPenalty += penalty
        print("Test return is ",totalReward/(nb))
        print("Test Penalty is ",totalPenalty/(nb))