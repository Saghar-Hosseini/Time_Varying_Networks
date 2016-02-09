import Config
def OpinionDynamics(mu,node,neighbors):
    '''
     :param mu:  convegence coeeficient in opinion dynamics
     :param node: a node in the network
     :param neighbors: the list neighbors of node including itslef
     :return: the updated state of node based on opinion dynamics
    '''
    numCommunity=Config.numCommunity
    FJ=False
    CB=True
    next_state=[0]*numCommunity
    N=len(neighbors)
    #Friedkin_Johnsen method
    if FJ:
        for d in range(numCommunity):
            next_state[d]=mu*Config.initial_state[node][d]
        for n in range(len(neighbors)):
            for d in range(numCommunity):
                next_state[d]+=(1-mu)*Config.state[neighbors[n]][d]
    if CB:
        for n in range(1,N):
            for d in range(numCommunity):
                next_state[d]+=mu*Config.state[neighbors[n]][d]/float(N-1)
        for d in range(numCommunity):
            next_state[d]+=(1-mu)*Config.state[node][d]
    return next_state

