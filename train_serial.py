s__author__ = 'saghar hosseini (saghar@uw.edu)'
from ReadData import *
import numpy as np
from Sample import*
import csv
from write_data import*
from projection import*
from opinion_dynamics import*
from gradient_calculation import*
import Config

############################################################################################
#                               Define Parameters
############################################################################################
numberOfSnapshots=Config.numberOfSnapshots
numCommunity=Config.numCommunity
mu=Config.mu
lambdah_C=Config.lambdah_C
lambdah_B=Config.lambdah_B
sampleFraction=Config.sampleFraction
K_B=Config.K_B
K_C=Config.K_C
path=Config.path
output_path=Config.output_path
gradients={}
############################################################################################
#                           Save Initial Results
############################################################################################
print 'iteration 0'
output_file=output_path+'state_output'+str(0)+'.csv'
write_to_cvs(output_file,Config.state,numCommunity)
# output_file=path+'running_avg_state_output'+str(t)+'.csv'
# write_to_cvs(output_file,running_avg_state,numCommunity)
output_file=output_path+'B_output'+str(0)+'.csv'
np.savetxt(output_file, Config.B, delimiter=",")
# output_file=path+'running_avg_B_output'+str(t)+'.csv'
# np.savetxt(output_file, running_avg_B, delimiter=",")
############################################################################################
#                                   Training
############################################################################################
gradient_norm=0.0
for t in range(2,numberOfSnapshots+1):
    # learning_rate=1.0/(np.sqrt(t))
    # learning_rate=1.0
    for iter in range(1,2):
        #learning_rate=np.sqrt(2.0/(iter+1))
        # draw a set of random nodes
        if Config.n >= 1000:
            numSamples=int(sampleFraction*Config.n)
        else:
            numSamples=Config.n
        sampleSet=Sample({},{},[])
        sampleSet.NodeSampling(numSamples)
        ##################################################################
        #*****************    update states     *************************
        for i in sampleSet.nodes:
            if not i in Config.state.keys():
                # Config.initial_state[i]=np.array([1.0/numCommunity]*numCommunity)
                Config.initial_state[i]=np.random.random_sample((numCommunity,))
                Config.initial_state[i]/=np.sum(Config.initial_state[i])
                Config.state[i]=Config.initial_state[i]
            Config.visit[i]=0.0
        for i in sampleSet.nodes:
            #Opinion Dynamics
            state_OPD=OpinionDynamics(mu,i,sampleSet.links[i])
            #calculate gradient for both links and non-links
            gradients[i]=calculate_gradient_state(i,lambdah_C,sampleSet)
            Config.visit[i]+=np.linalg.norm(gradients[i])
            Config.learning_rate_C[i]=K_C/Config.visit[i]
        for i in sampleSet.nodes:
            #update state of node i based on mirror descent
            # new_state=state_OPD+Config.learning_rate_C[i]*gradients[i]
            # new_state=Config.state[i]-Config.learning_rate_C[i]*(Config.state[i]-state_OPD)-Config.learning_rate_C[i]*(1.0-Config.state[i])*gradients[i]
            new_state=Config.state[i]-Config.learning_rate_C[i]*gradients[i] # without opinion dynamics
            #project the state into the simplex
            Config.state[i]=projection_simplex(new_state, 1.0)
        ###################################################################
        #*************         update community matrix B       ************
        #gradient_B=np.zeros((numCommunity,numCommunity))
        gradient_B=calculate_gradient_corelation_matrix(lambdah_B,sampleSet)
        gradient_norm+=np.linalg.norm(gradient_B,ord=2)
        # update matrix B based on the sub-gradient descent step
        Config.B-=gradient_B/gradient_norm
        # project the matrix B into the stochastic matrices space
        # for k in range(numCommunity):
        #     B[k]=projection_simplex(B[k],1.0)
            #count+=1
        # Config.B=Doubly_Stochastic_Normalization(Config.B,1000)
        Project_B()
    #########################################################################
    #*****************         Load next Graph        ***********************
    Config.edges=Config.dataset.read_network_snapshot(t,hasHeader=True)
    Config.nodes_list=Config.nodes_list.union(Config.edges.keys())
    Config.n=len(Config.edges.keys())
    #########################################################################
    #*****************          Save results          ***********************
    print 'Iteration '+str(t)
    output_file=output_path+'state_output'+str(t)+'K='+str(numCommunity)+'.csv'
    write_to_cvs(output_file,Config.state,numCommunity)
    output_file=output_path+'B_output'+str(t)+'K='+str(numCommunity)+'.csv'
    np.savetxt(output_file, Config.B, delimiter=",")

