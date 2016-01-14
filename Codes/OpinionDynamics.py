
def OpinionDynamics(initial_state,state,mu,node,neighbors,numCommunity):
   '''
    :param mu:  convegence coeeficient in opinion dynamics
    :param node: a node in the network
    :param neighbors: the list neighbors of node including itslef
    :return: the updated state of node based on opinion dynamics
   '''
   #Friedkin_Johnsen method
   next_state=[0]*numCommunity
   for d in range(numCommunity):
        next_state[d]=mu*initial_state[node][d]
   for n in range(len(neighbors)):
        if not neighbors[n] in state.keys():
            initial_state[neighbors[n]]=np.random.random_sample((numCommunity,))
            initial_state[neighbors[n]]/=np.sum(initial_state[neighbors[n]])
            state[neighbors[n]]=initial_state[neighbors[n]]
        for d in range(numCommunity):
            next_state[d]+=(1-mu)*state[neighbors[n]][d]
   return next_state
