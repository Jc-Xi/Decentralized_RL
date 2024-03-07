import numpy as np
import dill as pickle
from sklearn.preprocessing import PolynomialFeatures
from Bus import load_bus
from Line import load_line
from LowLevelEnv import LowLevelEnvironment as LLE
from UpLevelEnv import UpLevelEnvironment as ULE
from ActorCritic import ActorCritic as AC
from ScenarioGeneration import generate_scenario
from evaluation import eval
if __name__ == '__main__':
    with open('Lines_data.pkl', 'rb') as inp:
        LineConstraints = pickle.load(inp)
    with open('Buses_data.pkl', 'rb') as inp:
        BusConstraints = pickle.load(inp)
    nb_interval = 24
    
    Buses= {1:{},2:{},3:{}}
    bus2area = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}
    boundaries = {1:[2,3],2:[4,6],3:[7,8]}
    load_bus(BusConstraints,Buses,bus2area,nb_interval=nb_interval)

    Lines = {'Tie':{},'Trans':{1:{},2:{},3:{}}}
    load_line(LineConstraints,Lines,Buses,bus2area)

    LENV = dict()
    for area in Buses.keys():
        LENV[area] = LLE(Buses[area],Lines['Trans'][area],nb_interval)
    UENV = dict()
    for key,val in Lines['Tie'].items():
        UENV[key] = ULE(val,nb_interval)
    load_scale = 0.005
    poly = PolynomialFeatures(2)
    l_feature = {area: poly.fit_transform(np.array(env.get_obs().tolist()+[0,0]).reshape(1,-1)).shape[1]  for area,env in LENV.items() }#2 for the trans out of the boundary nodes
    l_action =  {area: env.act_dim for area,env in LENV.items()}
    u_feature = {line: poly.fit_transform(env.get_obs().reshape(1,-1)).shape[1] for line,env in UENV.items()}
    u_action =  {line: 1 for line in UENV.keys()}


    np.random.seed(20)
    AREA_AGENT = dict()
    for area in LENV.keys():
        AREA_AGENT[area] = AC(nb_interval,l_feature[area],l_action[area])
    
    LINE_AGENT = dict()
    for line in UENV.keys():
        LINE_AGENT[line] = AC(nb_interval,u_feature[line],u_action[line])

    error_sce = generate_scenario(Buses)

    
    #upper_step
    def upper_step(agent,env,t):
        obs = env.get_obs()
        obs = poly.fit_transform(obs.reshape(1,-1))[0]
        act = agent.actor_dict[t](obs)
        env.apply_action(act)
        
    #lower_step
    def lower_step(area,agent,env,t,error = {}):
        obs = env.get_obs()
        b1,b2 = boundaries[area]
        obs = np.hstack((obs,np.array([env.buses[b1].trans_out*load_scale,env.buses[b2].trans_out*load_scale])))
        obs = poly.fit_transform(obs.reshape(1,-1))[0]
        act = agent.actor_dict[t](obs)
        env.step(act,error)
    
    
    #Training
    epoch  = 0
    inner_training = 20
    outer_training = 5
    for agent in AREA_AGENT.values():
        agent.update_sigma(0.3)
    for agent in LINE_AGENT.values():
        agent.update_sigma(0.01)
    for env in LENV.values():
        env.train()

    while epoch <50_000:
        #lower
        for k in range(inner_training):
            for agent in AREA_AGENT.values():
                agent.decay_sigma(0.999999)
            for env in LENV.values():
                env.reset()
            for env in UENV.values():
                env.reset()
            obs_dict = { area:env.get_obs() for area,env in LENV.items()}
            for line in UENV.keys():
                upper_step(LINE_AGENT[line],UENV[line],0)
                src,end = line
                obs_dict[bus2area[src]] = np.hstack((obs_dict[bus2area[src]],np.array([Buses[bus2area[src]][src].trans_out*load_scale])))
                obs_dict[bus2area[end]] = np.hstack((obs_dict[bus2area[end]],np.array([Buses[bus2area[end]][end].trans_out*load_scale])))
            for key,val in obs_dict.items():
                obs_dict[key] = poly.fit_transform(val.reshape(1,-1))[0]
            for t in range(nb_interval):
                grad_dict = {}
                new_obs_dict = {}
                reward_dict = {}
                penalty_dict = {}
                for area,agent in AREA_AGENT.items():
                    act,grad_dict[area] = agent.select_action(obs_dict[area],t)
                    new_obs_dict[area],reward_dict[area],penalty_dict[area] = LENV[area].step(act)
                for line in UENV.keys():
                    UENV[line].step()
                if t <  nb_interval-1: #new_obs has no effect if t = 23
                    for line in UENV.keys():
                        upper_step(LINE_AGENT[line],UENV[line],t+1)
                        src,end = line
                        new_obs_dict[bus2area[src]] = np.hstack((new_obs_dict[bus2area[src]],np.array([Buses[bus2area[src]][src].trans_out*load_scale])))
                        new_obs_dict[bus2area[end]] = np.hstack((new_obs_dict[bus2area[end]],np.array([Buses[bus2area[end]][end].trans_out*load_scale])))
                    for key,val in new_obs_dict.items():
                        new_obs_dict[key] = poly.fit_transform(val.reshape(1,-1))[0]
                for area,agent in AREA_AGENT.items():
                    agent.update_policy(obs_dict[area],grad_dict[area],reward_dict[area]+penalty_dict[area],\
                                        new_obs_dict[area],t,1e-4,1e-3)
                obs_dict = new_obs_dict
        #upper
        for k in range(outer_training):
            for agent in LINE_AGENT.values():
                agent.decay_sigma(0.999999)
            for env in LENV.values():
                env.reset()
            for env in UENV.values():
                env.reset()  
            for t in range(nb_interval):
                obs_dict = { line:poly.fit_transform(env.get_obs().reshape(1,-1))[0] for line,env in UENV.items()}
                grad_dict = {}
                new_obs_dict = {}
                reward_dict = {}
                for line,agent in LINE_AGENT.items():
                    act,grad_dict[line] = agent.select_action(obs_dict[line],t)
                    UENV[line].apply_action(act)
                for area in LENV.keys():
                    lower_step(area,AREA_AGENT[area],LENV[area],t)
                for line in UENV.keys():
                    new_obs_dict[line],reward_dict[line] = UENV[line].step()
                    new_obs_dict[line] = poly.fit_transform(new_obs_dict[line].reshape(1,-1))[0]
                    LINE_AGENT[line].update_policy(obs_dict[line],grad_dict[line],reward_dict[line],\
                                                   new_obs_dict[line],t,1e-4,1e-3)
        if (epoch+1)%100 ==0:
            print("=============EPOCH {}===============".format(epoch))
            for env in LENV.values():
                env.evaluate()
            eval(error_sce,poly,AREA_AGENT,LINE_AGENT,LENV,UENV,boundaries)
            for env in LENV.values():
                env.train()
        epoch +=1

                    


