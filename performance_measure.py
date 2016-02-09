__author__ = 'sagha_000'
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import csv
from ReadData import *
import pandas as pd
import Config
#######################################################################################
def read_in_states(filename, has_header):
    state={}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 1 and has_header:
                continue
            fields = line.split(",")
            node = fields[0]
            if not state.has_key(node):
                state[node] = np.array(map(float,fields[1:]))
    return state
###################################################################################

def processInput(T,i,state_sum,B_sum,state,B,edges,nodes_list):
    import Config
    B=np.matrix(B)
    B_avg=np.matrix(B_sum)/T
    error=0.0
    error_avg=0.0
    if  i not in state.keys():
        # initilize new node's state
        #*** uniform distribution ******
        # state[i]=np.array([1.0/numCommunity]*numCommunity)
        #*** random distribution ******
        state[i]=np.random.random_sample((Config.numCommunity,))
        state[i]/=np.sum(state[i])
        # update the running sum of states
        Config.state_sum[i]=np.copy(state[i])
        state_sum[i]=np.copy(Config.state_sum[i])
    else:
        state_sum[i]/=T
    for j in nodes_list:
        if j not in state.keys():
            # initilize new node's state
            #*** uniform distribution ******
            # state[j]=np.array([1.0/numCommunity]*numCommunity)
            #*** random distribution ******
            state[j]=np.random.random_sample((Config.numCommunity,))
            state[j]/=np.sum(state[j])
            # update the running sum of states
            state_sum[j]=np.copy(state[j])
            Config.state_sum[j]=np.copy(state_sum[j])
        else:
            state_sum[j]/=T
        xi_avg=np.matrix(state_sum[i])
        xj_avg=np.matrix(state_sum[j])
        xi=np.matrix(state[i])
        xj=np.matrix(state[j])
        # compute the probability of existing an edge
        prob_avg=xi_avg*B_avg*(xj_avg.T)
        prob=xi*B*(xj.T)
        estimate=np.rint(prob.item(0))
        estimate_avg=np.rint(prob_avg.item(0))
        if (i in edges.keys() and j in edges[i]) or j==i:
            error+=abs(1.0-estimate)
            error_avg+=abs(1.0-estimate_avg)
        else:
            error+=abs(0.0-estimate)
            error_avg+=abs(0.0-estimate_avg)
    return error,error_avg

#########################################################################################
if __name__ == '__main__':
    path="F:/Saghar_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
    # path="C:/Users/sagha_000/Documents/SVN/My_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
    Config.dataset=ReadData(path)
    numberOfSnapshots=Config.numberOfSnapshots
    numCommunity=Config.numCommunity
    error=[0.0]*numberOfSnapshots
    error_avg=error[:]
    error_file=path+'error_output.csv'
    test_array = []
    num_cores = multiprocessing.cpu_count()
    for t in range(numberOfSnapshots):
        Config.edges=Config.dataset.read_network_snapshot(t)
        filename=path+'state_output'+str(t)+'.csv'
        Config.state=read_in_states(filename, has_header=True)
        Config.nodes_list=set(Config.edges.keys())
        Config.nodes_list.union(Config.state.keys())
        #make it parallel
        for i in Config.state.keys():
              if not i in Config.state_sum.keys():
                  Config.state_sum[i]=np.copy(Config.state[i])
              else:
                  for c in range(numCommunity):
                      Config.state_sum[i][c]+=Config.state[i][c]
        filename=path+'B_output'+str(t)+'.csv'
        df=pd.read_csv(filename, sep=',',header=None)
        Config.B=df.values
        Config.B_sum+=Config.B

        results = Parallel(n_jobs=num_cores)(delayed(processInput)(t+1,i,Config.state_sum,Config.B_sum,Config.state,Config.B,Config.edges,Config.nodes_list) for i in Config.nodes_list)
        for i in range(len(results)):
            error[t]+=results[i][0]
            error_avg[t]+=results[i][1]
        total_nodes=len(Config.nodes_list)
        print 'True error='+ str(error[t])+'for t='+str(t)
        print 'True avg error='+ str(error_avg[t])+'for t='+str(t)
        print 'number of edges= '+str(len(Config.edges))
        fld=['time','error','avg error']
        file = open(error_file,'wb')
        csvwriter = csv.DictWriter(file, delimiter=',', fieldnames=fld)
        csvwriter.writerow(dict((fn,fn) for fn in fld))
        row={}
        for k in range(len(error)):
            row['time']=k
            row['error']=error[k]
            row['avg error']=error_avg[k]
            csvwriter.writerow(row)
        file.close()