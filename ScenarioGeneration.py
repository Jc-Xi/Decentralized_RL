import numpy as np
def generate_scenario(Buses,nb_scenarios =10000,seed=  0,nb_interval = 24):
    np.random.seed(0)
    nb_bus_ = 3
    error= dict()
    error['area1'] = np.zeros((nb_scenarios,24,nb_bus_))
    error['area2'] = np.zeros((nb_scenarios,24,nb_bus_))
    error['area3'] = np.zeros((nb_scenarios,24,nb_bus_))
    for i in range(9):
        error['area1'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
        error['area2'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
        error['area3'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
    for i in range(17,24):
        error['area1'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
        error['area2'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
        error['area3'][:,i] = np.clip(np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),-100,100)
    for i in range(9,17):
        error['area1'][:,i] = np.clip(error['area1'][:,i-1] + 0.5*np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),
                                    -100,100)
        error['area2'][:,i] = np.clip(error['area2'][:,i-1] + 0.5*np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),
                                    -100,100)
        error['area3'][:,i] = np.clip(error['area3'][:,i-1] + 0.5*np.random.normal(0,10,size = (nb_scenarios,nb_bus_)),
                                    -100,100)
    error_sce = dict()
    for sce in range(nb_scenarios):
        error_sce[sce] = {}
        for t in range(nb_interval):
            error_sce[sce][t] = {}
            for area in Buses.values():
                for index,bus in area.items():
                    if bus.perturbation:
                        if index in range(1,4):
                            error_sce[sce][t][index] = error['area1'][sce][t][index-1]
                        elif index in range(4,7):
                            error_sce[sce][t][index] = error['area2'][sce][t][index-nb_bus_-1]
                        elif index in range(7,10):
                            error_sce[sce][t][index] = error['area3'][sce][t][index-2*nb_bus_-1]
    return error_sce