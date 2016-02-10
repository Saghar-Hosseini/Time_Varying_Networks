__author__ = 'saghar hosseini (saghar@uw.edu)'
import numpy as np
from ReadData import*
from projection import*
##########################################################################################
#                              Load Data
##########################################################################################
# path="C:/Users/sagha_000/Documents/SVN/My_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
path="F:/Saghar_SVN/TimeVaryingSocialNetworks/datasets/twitter-pol-dataset/graphs/"
dataset=ReadData(path)
edges=dataset.read_network_snapshot(1,hasHeader=True)
nodes_list=set(edges.keys())
output_path='F:/Saghar_SVN/TimeVaryingSocialNetworks/datasets/twitter-pol-dataset/Results/wo_OPD/'
############################################################################################
#                               Define Parameters
############################################################################################
numberOfSnapshots=1175
numCommunity=10
mu=0.1
lambdah_C=0.0
lambdah_B=0.0
sampleFraction=0.25
n=len(nodes_list)
K_B=1.0
K_C=1.0
#############################################################################################
#variables
learning_rate_C={}
initial_state=dict()
state=dict()
visit={}
state_sum={}

###########################################################################################
#**************           initialize the variables             ****************************
###########################################################################################
B=np.zeros((numCommunity,numCommunity))
B_sum=np.zeros((numCommunity,numCommunity))
for node in nodes_list:
    # initial_state[node]=np.array([1.0/numCommunity]*numCommunity)
    initial_state[node]=np.random.random_sample((numCommunity,))
    initial_state[node]/=np.sum(initial_state[node])
    state[node]=np.copy(initial_state[node])
    state_sum[node]=np.array([0.0]*numCommunity)
for c in range(numCommunity):
#     B[c]=np.random.random_sample((numCommunity,))
    B[c]=np.array([0.5]*numCommunity)
#     B[c][c]=1.0
#     B[c]/=np.sum(abs(B[c]))
# B=Doubly_Stochastic_Normalization(B,1000)
# B=np.identity(numCommunity)




