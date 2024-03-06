from utils import *
class Line(object): #including both tieline and transmission line
    def __init__(self,constraints,Buses,bus2area,nb_intervals = 24,rho = 0.05):
        self.nodeSource = Buses[bus2area[constraints['nodeSource']]][constraints['nodeSource']]
        self.nodeEnd = Buses[bus2area[constraints['nodeEnd']]][constraints['nodeEnd']]
        self.max_trans = constraints['max_trans']
        self.avg_flow = np.zeros(nb_intervals)
        self.rho = rho
    def rescale(self,flow_action,t=0, trans = True):
        if trans:
            return rescale_action(flow_action,-self.max_trans,self.max_trans)
        else:
            return np.clip(rescale_action(flow_action,-self.max_trans,self.max_trans), self.avg_flow[t]-0.1*self.max_trans,self.avg_flow[t]+0.1*self.max_trans)
    def update_transout(self,flow_action):
        self.nodeSource.update_transout(flow_action)
        self.nodeEnd.update_transout(-flow_action)
    def update_source(self,flow_action):
        self.nodeSource.update_transout(flow_action)
    def update_end(self,flow_action):
        self.nodeEnd.update_transout(-flow_action)
    def penalty(self,flow_action,coef,t):
        return coef*(flow_action-self.avg_flow[t]) + self.rho*np.square(flow_action-self.avg_flow[t])
    def update_avg_flow(self,flow,t):
        self.avg_flow[t] = flow
def load_line(constraints,Lines,Buses,bus2area,rho = 0.01):
    for key,val in constraints.items():
        node1,node2 = key
        if bus2area[node1] == bus2area[node2]:
            Lines['Trans'][bus2area[node1]][key] = Line(val,Buses,bus2area)
        else:
            Lines['Tie'][key] = Line(val,Buses,bus2area,rho = rho)