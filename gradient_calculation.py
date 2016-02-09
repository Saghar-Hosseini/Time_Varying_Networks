import numpy as np
import Config
def calculate_gradient_state(node,lambdah,sampleSet):
    '''
    :param node: a node in the network
    :param B: Community matrix
    :param lambdah: regularizer coefficient
    :return: the sub-gradient with respect to the state of node
    '''
    # Calculate the gradient
    i=node
    B=np.matrix(Config.B)
    xi=np.matrix(Config.state[i])
    numSamples=len(sampleSet.nodes)
    numPairs=numSamples*(numSamples-1)/2.0
    gradient=lambdah*xi.T
    for k in range(len(sampleSet.links[i])):
        j=sampleSet.links[i][k]
        xj=np.matrix(Config.state[j])
        diff=1.0/sampleSet.pdf_links[i][k]/numPairs*(1.0-xi*B*(xj.T))
        if j==i:
            gradient-=diff.item(0)*(B+B.T)*(xi.T)
        else:
            gradient-=diff.item(0)*B*(xj.T)
    for k in range(len(sampleSet.non_links[i])):
        j=sampleSet.non_links[i][k]
        diff=1.0/sampleSet.pdf_non_links[i][k]/numPairs*(0.0-xi*B*(xj.T))
        gradient-=diff.item(0)*B*(xj.T)
    return np.squeeze(np.asarray(gradient))

################################################################################3

def calculate_gradient_corelation_matrix(lambdah,sampleSet):
    '''
    :param sampleSet:  set of nodes used for updateing the matrix B
    :param non_links: list of non-links associated with the nodes in the sample set
    :param B: correlation/community matrix
    :param lambdah: regularizer coefficient
    :return: the sub-gradient of cost function w.r.t. matrix B
    '''
    numCommunity=Config.numCommunity
    B=np.matrix(Config.B)
    gradient_B=lambdah*B
    numSamples=len(sampleSet.nodes)
    numPairs=numSamples**2
    for i in sampleSet.nodes:
        xi=np.matrix(Config.state[i])
        for k in range(len(sampleSet.links[i])):
            j=sampleSet.links[i][k]
            xj=np.matrix(Config.state[j])
            diff=1.0/sampleSet.pdf_links[i][k]/numPairs*(1.0-xi*B*(xj.T))
            gradient_B-=diff.item(0)*(xi.T)*xj
        for k in range(len(sampleSet.non_links[i])):
            j=sampleSet.non_links[i][k]
            diff=1.0/sampleSet.pdf_non_links[i][k]/numPairs*(0.0-xi*B*(xj.T))
            gradient_B-=diff.item(0)*(xi.T)*xj
    return gradient_B
