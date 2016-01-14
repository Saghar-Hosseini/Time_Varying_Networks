__author__ = 'sagha_000'
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import csv
from ReadData import *
import pandas as pd
#######################################################################################
def read_in_states(filename, has_header):
    state={}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i < 2 and has_header:
                continue
            fields = line.split(",")
            node = int(fields[0])
            if not state.has_key(node):
                state[node] = np.array(fields[1:])
    return state
###################################################################################

def processInput(i,state,B,numCommunity,edges,nodes_list):
    error=0
    #new node
    if not i in state.keys():
        state[i]=np.random.random_sample((numCommunity,))
        state[i]/=np.sum(state[i])
    for j in nodes_list:
        if not j in state.keys():
            state[j]=np.random.random_sample((numCommunity,))
            state[j]/=np.sum(state[j])
        if i in edges.keys() and (j in edges[i] or j==i):
            error+=abs(1-(state[i].dot(B)).dot(state[j]))
        else:
            error+=abs(0-(state[i].dot(B)).dot(state[j]))
    return error

#########################################################################################
#path="C:/Users/sagha_000/Documents/SVN/My_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
if __name__ == '__main__':
    path="F:/Saghar_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
    dataset=ReadData(path)
    numberOfSnapshots=300
    numCommunity=10
    error=[0.0]*numberOfSnapshots
    error_file=path+'error_output.csv'
    test_array = []
    num_cores = multiprocessing.cpu_count()
    for t in range(numberOfSnapshots):
        edges=dataset.read_network_snapshot(t)
        nodes_list=edges.keys()
        filename=path+'state_output'+str(t)+'.csv'
        state=read_in_states(filename, has_header=True)
        filename=path+'B_output'+str(t)+'.csv'
        df=pd.read_csv(filename, sep=',',header=None)
        B=df.values
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,state,B,numCommunity,edges,nodes_list) for i in nodes_list)
        error[t]=sum(results)
        print 'True error='+ str(error[t])+'for t='+str(t)
        test_array.append({'error':error[t]})
        test_array.append({'time':t})
        test_file = open(error_file,'wb')
        csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=['time','error'])
        for row in test_array:
            csvwriter.writerow(row)
        test_file.close()


# avg_state={}
# for t in range(numberOfSnapshots):
#     edges=dataset.read_network_snapshot(t)
#     nodes_list=edges.keys()
#     for node in nodes_list:
#         avg_state[node]=[0.0]*numCommunity
#     for k in range(t):
#         filename=path+'state_output'+str(k)+'.csv'
#         state=read_in_states(filename, has_header=True)
#         filename=path+'B_output'+str(k)+'.csv'
#         df=pd.read_csv(filename, sep=',',header=None)
#         B=df.values
#         for node in state.keys():
#             avg_state[node]+=state[node]
#     for node in state.keys():
#             avg_state[node]=1/(t+1)*avg_state[node]
#
#
