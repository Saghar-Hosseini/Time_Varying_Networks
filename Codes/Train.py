s__author__ = 'saghar'
from ReadData import *
#import random
import numpy as np
from Sample import*
import csv

############################################################
def test_error(avg):
    err=0
    if avg:
        for i in nodes_list:
            for j in nodes_list:
                #new node
                if not i in avg_state.keys():
                   initial_state[i]=np.random.random_sample((numCommunity,))
                   initial_state[i]/=np.sum(initial_state[i])
                   state[i]=initial_state[i]
                   avg_state[i]=state[i]
                   state_counter[i]=0.0
                if not j in avg_state.keys():
                   initial_state[j]=np.random.random_sample((numCommunity,))
                   initial_state[j]/=np.sum(initial_state[j])
                   state[j]=initial_state[j]
                   avg_state[j]=state[j]
                   state_counter[j]=0.0
                if i in edges.keys():
                    if j in edges[i] or j==i:
                         err+=abs(1-(avg_state[i].dot(B)).dot(avg_state[j]))
                    else:
                        err+=abs(0-(avg_state[i].dot(B)).dot(avg_state[j]))
                else:
                    err+=abs(0-(avg_state[i].dot(B)).dot(avg_state[j]))
    else:
        for i in nodes_list:
            for j in nodes_list:
                #new node
                if not i in state.keys():
                   initial_state[i]=np.random.random_sample((numCommunity,))
                   initial_state[i]/=np.sum(initial_state[i])
                   state[i]=initial_state[i]
                   avg_state[i]=state[i]
                   state_counter[i]=0.0
                if not j in state.keys():
                   initial_state[j]=np.random.random_sample((numCommunity,))
                   initial_state[j]/=np.sum(initial_state[j])
                   state[j]=initial_state[j]
                   avg_state[j]=state[j]
                   state_counter[j]=0.0
                if i in edges.keys():
                    if j in edges[i] or j==i:
                         err+=abs(1-(state[i].dot(B)).dot(state[j]))
                    else:
                        err+=abs(0-(state[i].dot(B)).dot(state[j]))
                else:
                    err+=abs(0-(state[i].dot(B)).dot(state[j]))
    return err

#################################################################################################
def OpinionDynamics(mu,node,neighbors):
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
        for d in range(numCommunity):
            next_state[d]+=(1-mu)*state[neighbors[n]][d]
   return next_state

##################################################################################

def calculate_gradient_state(node,lambdah,sampleSet):
    '''
    :param node: a node in the network
    :param B: Community matrix
    :param lambdah: regularizer coefficient
    :return: the sub-gradient with respect to the state of node
    '''
    # Calculate the gradient
    i=node
    gradient=lambdah*np.array(state[i])
    for k in range(len(sampleSet.links[i])):
        j=sampleSet.links[i][k]
        diff=1.0/sampleSet.pdf_links[i][k]/numSamples*(1-(state[i].dot(B)).dot(state[j]))
        gradient+=-2*diff*state[j].dot(B)
    for k in range(len(sampleSet.none_links[i])):
        j=sampleSet.none_links[i][k]
        diff=1.0/sampleSet.pdf_non_links[i][k]/numSamples*(0-(state[i].dot(B)).dot(state[j]))
        gradient+=-2*diff*state[j].dot(B)
    return gradient

################################################################################3

def calculate_gradient_corelation_matrix(lambdah,sampleSet):
    '''
    :param sampleSet:  set of nodes used for updateing the matrix B
    :param non_links: list of non-links associated with the nodes in the sample set
    :param B: correlation/community matrix
    :param lambdah: regularizer coefficient
    :return: the sub-gradient of cost function w.r.t. matrix B
    '''
    gradient_B=lambdah*B
    for i in sampleSet.nodes:
        for k in range(len(sampleSet.links[i])):
            j=sampleSet.links[i][k]
            diff=1.0/sampleSet.pdf_links[i][k]/numSamples*(1-(state[i].dot(B)).dot(state[j]))
            gradient_B+=-2*diff*np.dot(state[j],state[i].T)
        for k in range(len(sampleSet.none_links[i])):
            j=sampleSet.none_links[i][k]
            diff=1.0/sampleSet.pdf_non_links[i][k]/numSamples*(0-(state[i].dot(B)).dot(state[j]))
            gradient_B+=-2*diff*np.dot(state[j],state[i].T)
    return gradient_B

#####################################################################################
# Data-set
path="C:/Users/sagha_000/Documents/SVN/My_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
#path="F:/Saghar_SVN/TimeVaryingSocialNetworks/datasets/as-733/"
dataset=ReadData(path)
numberOfSnapshots=733

#define parameters
initial_state={}
state={}
new_state={}
#avg_state={}
numCommunity=10
B=np.zeros((numCommunity,numCommunity))
#avg_B=np.zeros((numCommunity,numCommunity))
mu=0.8
lambdah_C=0.00000001
lambdah_B=0.00000001
sampleFraction=0.05
nodes_list=set()
error=[0.0]*numberOfSnapshots
error_avg=[0.0]*numberOfSnapshots
#state_counter={}
#************** initialize the variables ****************************
edges=dataset.read_network_snapshot(0)
nodes_list=nodes_list.union(edges.keys())
t=0
for node in nodes_list:
    #initial_state[node]=np.array([1.0/numCommunity]*numCommunity)
    initial_state[node]=np.random.random_sample((numCommunity,))
    initial_state[node]/=np.sum(initial_state[node])
    state[node]=initial_state[node]
    #avg_state[node]=state[node]
    #state_counter[node]=0.0
for c in range(numCommunity):
    B[c]=np.random.random_sample((numCommunity,))
    B[c][c]=1.0
    B[c]/=np.sum(abs(B[c]))
    #avg_B[c]=B[c]
#error[0]=test_error(avg=False)
#error_avg[0]=test_error(avg=True)
print 'True error='+ str(error[0])+'for t=0'
#print 'True error for running average'+str(error[0])+'for t=0'
output_file=path+'state_output'+str(t)+'.csv'
test_file = open(output_file,'wb')
fld=['node']
fld.extend(range(numCommunity))
csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=fld)
csvwriter.writerow(dict((fn,fn) for fn in fld))
row={}
for node in state.keys():
        row['node']=node
        for i in range(numCommunity):
            row[i]=state[node][i]
        csvwriter.writerow(row)
test_file.close()
output_file=path+'B_output'+str(t)+'.csv'
np.savetxt(output_file, B, delimiter=",")

#**********************************************************************
############### Training #####################################
count=0.0
N=len(nodes_list)*numCommunity**2
for t in range(1,numberOfSnapshots):
    learning_rate=1.0/(t+N)
    for iter in range(100):
        #learning_rate=np.sqrt(2.0/(iter+1))
        # draw a set of random nodes
        numSamples=int(sampleFraction*len(nodes_list))
        sampleSet=Sample({},{},[])
        #sampleSet.NodeSampling(numSamples,edges,nodes_list)
        sampleSet.InducedEdgeSampling(numSamples,nodes_list,edges)
        ##################################################################
        # update states
        for i in sampleSet.nodes:
            if not i in state.keys():
               #initial_state[i]=np.array([1.0/numCommunity]*numCommunity)
               initial_state[i]=np.random.random_sample((numCommunity,))
               initial_state[i]/=np.sum(initial_state[i])
               state[i]=initial_state[i]
               #avg_state[i]=state[i]
               #state_counter[i]=0.0
        for i in sampleSet.nodes:
            #Opinion Dynamics
            new_state[i]=OpinionDynamics(mu,i,sampleSet.links[i])
            #calculate gradient for both links and non-links
            gradient=calculate_gradient_state(i,lambdah_C,sampleSet)
            #update state of node i based on mirror descent
            new_state[i]-=learning_rate*gradient
            #project the state into the simplex
            totWeight=np.sum(new_state[i])
            state[i]=new_state[i]/totWeight
            #avg_state[i]=1/(state_counter[i]+1)*(state_counter[i]+count*avg_state[i])
            #state_counter[i]+=1
        #update community matrix B
        #gradient_B=np.zeros((numCommunity,numCommunity))
        gradient_B=calculate_gradient_corelation_matrix(lambdah_B,sampleSet)
        # update matrix B based on the sub-gradient descent step
        B-=learning_rate*gradient_B
        # project the matrix B into the stochastic matrices space
        for k in range(numCommunity):
            #B[k]/=np.sum(abs(B[k]))
            B[k]=np.abs(B[k])
            #avg_B[k]=1/(count+1)*(B[k]+count*avg_B[k])
        count+=1
    #print B
    #print state
    # average solutions

    edges=dataset.read_network_snapshot(t)
    nodes_list=nodes_list.union(edges.keys())
    #error[t]=test_error(avg=False)
    #error_avg[t]=test_error(avg=True)
    print 'True error='+ str(error[t])+'for t='+str(t)
    #print 'True error for running average'+str(error_avg[t])+'for t='+str(t)
    output_file=path+'state_output'+str(t)+'.csv'
    test_file = open(output_file,'wb')
    fld=['node']
    fld.extend(range(numCommunity))
    csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=fld)
    csvwriter.writerow(dict((fn,fn) for fn in fld))
    row={}
    for node in state.keys():
        row['node']=node
        for i in range(numCommunity):
            row[i]=state[node][i]
        csvwriter.writerow(row)
    test_file.close()
    output_file=path+'B_output'+str(t)+'.csv'
    np.savetxt(output_file, B, delimiter=",")


