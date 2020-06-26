# -*- coding: utf-8 -*-

from .MPSClass import TN
import cupy as cp

class PositiveMPS(TN):
    """Matrix Product States with non-negative parameters
    Parametrization using the square of real parameters.
    Parameters
    ----------
    D : int, optional
        Rank/Bond dimension of the MPS
    learning_rate : float, optional
        Learning rate of the gradient descent algorithm
    batch_size : int, optional
        Number of examples per minibatch.
    n_iter : int, optional
        Number of iterations (epochs) over the training dataset to perform
        during training.
    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    ----------
    Attributes
    ----------
    w : numpy array, shape (m_parameters)
        Parameters of the tensor network
    norm : float
        normalization constant for the probability distribution
    n_samples : int
        number of training samples
    n_features : int
        number of features in the dataset
    d : int
        physical dimension (dimension of the features)
    m_parameters : int
        number of parameters in the network
    history : list
        saves the training accuracies during training
    """
    def __init__(self, D=4, learning_rate=0.1, batch_size=10,
                 n_iter=100, random_state=None, verbose=False):
        self.D = D
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        
    def _probability(self, x):
        """Unnormalized probability of one configuration P(x)
        Parameters
        ----------
        x : cupy array, shape (n_features,)
            One configuration
        Returns
        -------
        probability : float
        """
        w2 = cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        tmp = cp.square(w2[0,x[0],0,:]) #First tensor
        for i in range(1,self.n_features-1):
            tmp = cp.dot(tmp,cupy.square(w2[i,x[i],:,:])) #MPS contraction  
        probability = cp.inner(tmp,cupy.square(w2[self.n_features-1,
                                                x[self.n_features-1],:,0]))
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        w2 = cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        tmp = cp.sum(cp.square(w2[0,:,0,:]),0) #First tensor
        for i in range(1,self.n_features-1):
            tmp = cp.dot(tmp,cp.sum(cp.square(w2[i,:,:,:]),0)) #MPS contraction  
        norm = cp.inner(tmp,cp.sum(cp.square(w2[self.n_features-1,:,:,0]),0))
        return norm
        
    def _derivative(self, x):
        """Compute the derivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative = cp.zeros((self.n_features,self.d,self.D,self.D))

        #Store intermediate tensor contractions for the derivatives: 
        #left to right and right to left
        #tmp stores the contraction of the first i+1 tensors from the left 
        #in tmp[i,:,:], tmp2 the remaining tensors on the right
        #the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp = cp.zeros((self.n_features,self.D))
        tmp2 = cp.zeros((self.n_features,self.D))
        tmp[0,:] = cp.square(w2[0,x[0],0,:])
        for i in range(1,self.n_features-1):
            tmp[i,:] = cp.dot(tmp[i-1,:],cp.square(w2[i,x[i],:,:]))  
        tmp[self.n_features-1,:] = cp.inner(tmp[self.n_features-2,:],
                cp.square(w2[self.n_features-1,x[self.n_features-1],:,0]))
        tmp2[self.n_features-1,:] = cp.square(w2[self.n_features-1,
                x[self.n_features-1],:,0])
        for i in range(self.n_features-2,-1,-1):
            tmp2[i,:] = cp.dot(cp.square(w2[i,x[i],:]),tmp2[i+1,:])
        tmp2[0,:] = cp.inner(cp.square(w2[0,x[0],0,:]),tmp2[1,:])
    
        #The derivative of each tensor is the contraction of the other tensors
        derivative[0,x[0],0,:] = cp.multiply(tmp2[1,:],2*(w2[0,x[0],0,:]))
        derivative[self.n_features-1,x[self.n_features-1],:,0] = \
                    cp.multiply(tmp[self.n_features-2,:],
                        2*(w2[self.n_features-1,x[self.n_features-1],:,0]))
        for i in range(1,self.n_features-1):
                derivative[i,x[i],:,:]=cp.multiply(cp.outer(tmp[i-1,:],
                tmp2[i+1,:]),2*(w2[i,x[i],:]))

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative = cp.zeros((self.n_features,self.d,self.D,self.D)) 
        
        tmp=cp.zeros((self.n_features,self.D))
        tmp2=cp.zeros((self.n_features,self.D))
        tmp[0,:]=cp.sum(cp.square(w2[0,:,0,:]),0)
        for i in range(1,self.n_features-1):
            tmp[i,:]=cp.dot(tmp[i-1,:],cp.sum(cp.square(w2[i,:,:,:]),0)) 
        tmp[self.n_features-1,:]=cp.inner(tmp[self.n_features-2,:],
                cp.sum(cp.square(w2[self.n_features-1,:,:,0]),0))
        tmp2[self.n_features-1,:]=cp.sum(cp.square(w2[self.n_features-1,:,:,0]),0)
        for i in range(self.n_features-2,-1,-1):
            tmp2[i,:]=cp.dot(cp.sum(cp.square(w2[i,:,:,:]),0),tmp2[i+1,:])
        tmp2[0,:]=cp.inner(cp.sum(cp.square(w2[0,:,0,:]),0),tmp2[1,:])
    
        for j in range(self.d):
            derivative[0,j,0,:]=cp.multiply(tmp2[1,:],2*(w2[0,j,0,:]))
            derivative[self.n_features-1,j,:,0]=\
                cp.multiply(tmp[self.n_features-2,:],2*(w2[self.n_features-1,j,:,0]))
        for i in range(1,self.n_features-1):
            temp3=cp.outer(tmp[i-1,:],tmp2[i+1,:])
            for j in range(self.d):
                derivative[i,j,:,:]=cp.multiply(temp3,2*(w2[i,j,:,:]))
        return derivative.reshape(self.m_parameters)

