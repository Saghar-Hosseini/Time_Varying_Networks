__author__ = 'sagha_000'
# based on http://www.magicbroom.info/Papers/DuchiShSiCh08.pdf
import numpy as np
import time
def project_onto_simplex(v,s=1):
    n=len(v)
    U=set(range(n))
    s=0.0
    rho=0.0
    u=np.zeros(n)
    while len(U)>0:
        G=set()
        L=set()
        k=np.random.choice(list(U),1)
        for i in U:
            if v[i]>=v[k]:
                G.add(i)
            else:
                L.add(i)
        delta_rho=len(G)
        delta_s=sum([v[i] for i in G])
        if s+delta_s-(rho+delta_rho)*v[k]<1:
            s+=delta_s
            rho+=delta_rho
            U=L
        else:
            G.discard(k[0])
            U=G
    theta=(s-1)/rho
    for i in range(n):
        u[i]=max(v[i]-theta,0)
    return u
################################################################################
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
####################################################################################
def projection_simplex(v, z=1):
    """
    Projection onto the simplex:
        w^* = argmin_w 0.5 ||w-v||^2 s.t. \sum_i w_i = z, w_i >= 0
    """
    # For other algorithms computing the same projection, see
    # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w
def projection_Birkhoff(M,x,y,error_tol,max_iter):
    k=0
    n=M.shape[1]
    zero=np.zeros([n,n])
    ones=np.ones([n,1])
    dual_gap=10
    while dual_gap > error_tol:
        Z=np.maximum(zero,np.dot(x,np.transpose(ones))+np.dot(ones,np.transpose(y))-M)
        x=1.0/n*(np.dot(M,ones)-(np.dot(np.transpose(y),ones)+1)*ones+np.dot(Z,ones))
        y=1.0/n*(np.dot(np.transpose(M),ones)-(np.dot(np.transpose(x),ones)+1)*ones+np.dot(np.transpose(Z),ones))
        primal=1.0/2*np.linalg.norm((Z-M),ord='fro')
        dual=-1.0/2*np.linalg.norm((np.dot(x,np.transpose(ones))+np.dot(ones,np.transpose(y))-Z),ord='fro')-\
             np.trace(np.dot(np.transpose(Z),M))+\
             np.dot(np.transpose(x),np.dot(M,ones)-ones)+np.dot(np.transpose(y),np.dot(np.transpose(M),ones)-ones)
        dual_gap=primal-dual
        k+=1
    return Z
def is_pos(x):
    return np.all(np.linalg.eigvals(x) >= 0)
def Doubly_Stochastic_Normalization(B,max_iter):
    X=np.copy(B)
    n=X.shape[1]
    I=np.identity(n)
    ones=np.ones([n,1])
    t=0
    while not is_pos(np.matrix(X)) and t < max_iter:
        X+=np.dot(1.0/n*(I-X)+1.0/n**2*np.dot(np.dot(np.transpose(ones),X),ones)*I,np.dot(ones,np.transpose(ones)))-\
           1.0/n*np.dot(np.dot(ones,np.transpose(ones)),X)
        t+=1
    if is_pos(X):
        return X
    else:
        return X.clip(0)
def Project_B():
    import Config
    for k in range(Config.numCommunity):
        for q in range(Config.numCommunity):
            if Config.B[k][q]> 1.0:
                Config.B[k][q]=1.0
            elif Config.B[k][q]<0.0:
                Config.B[k][q]=0.0
# start = time.clock()
# project_onto_simplex([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],1)
# print((time.clock() - start)*1000)
#
# start = time.clock()
# euclidean_proj_simplex(np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),1)
# print((time.clock() - start)*1000)
#
# start = time.clock()
# projection_simplex(np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),1)
# print((time.clock() - start)*1000)
# numCommunity=10
# B=np.zeros((numCommunity,numCommunity))
# for c in range(numCommunity):
#     B[c]=np.random.random_sample((numCommunity,))
#     # B[c]=np.array([0.0]*numCommunity)
#     B[c][c]=1.0
#     B[c]/=np.sum(abs(B[c]))
# x=np.ones([numCommunity,1])
# y=np.ones([numCommunity,1])
# projection_Birkhoff(B,x,y,1E-5,1000)
# D=Doubly_Stochastic_Normalization(B,1000)
# test1=np.dot(D,np.ones([numCommunity,1]))
# test2=np.dot(np.transpose(D),np.ones([numCommunity,1]))
# test3=D-np.transpose(D)
# test4=0
