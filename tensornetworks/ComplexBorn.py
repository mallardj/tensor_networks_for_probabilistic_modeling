# -*- coding: utf-8 -*-


from .MPSClass import TN
import numpy as cp


class ComplexBorn(TN):
    """Born machine with complex parameters
    Probability is the absolute value squared of the MPS
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
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        probability : float
        """
        w2 = cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        
        tmp = w2[0,x[0],0,:] #First tensor
        for i in range(1,self.n_features-1):
            tmp = cp.dot(tmp,w2[i,x[i],:,:]) #MPS contraction  
        output = cp.inner(tmp,w2[self.n_features-1,x[self.n_features-1],:,0])
        probability = cp.abs(output)**2
        return probability  

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        w2=cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        tmp = cp.tensordot(w2[0,:,0,:],cp.conj(w2[0,:,0,:]),
                           axes=([0],[0])).reshape(self.D*self.D)
        for i in range(1,self.n_features-1):
            tmp = cp.dot(tmp,cp.tensordot(w2[i,:,:,:],cp.conj(w2[i,:,:,:]),
                    axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,self.D*self.D)) 
        norm = cp.abs(cp.inner(tmp,cp.tensordot(w2[self.n_features-1,:,:,0],
                        cp.conj(w2[self.n_features-1,:,:,0]),
                        axes=([0],[0])).reshape(self.D*self.D)))
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
        w2=cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative=cp.zeros((self.n_features,self.d,self.D,self.D),dtype=cp.complex128)
        
        #Store intermediate tensor contractions for the derivatives: 
        #left to right and right to left
        #tmp stores the contraction of the first i+1 tensors from the left 
        #in tmp[i,:,:], tmp2 the remaining tensors on the right
        #the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp=cp.zeros((self.n_features,self.D),dtype=cp.complex128)
        tmp2=cp.zeros((self.n_features,self.D),dtype=cp.complex128)
        tmp[0,:]=w2[0,x[0],0,:]
        for i in range(1,self.n_features-1):
            tmp[i,:]=cp.dot(tmp[i-1,:],w2[i,x[i],:,:])  
        mpscontracted=cp.inner(tmp[self.n_features-2,:],w2[self.n_features-1,
                                       x[self.n_features-1],:,0])
        
        tmp[self.n_features-1,:]=cp.inner(tmp[self.n_features-2,:],
                        w2[self.n_features-1,x[self.n_features-1],:,0])
        tmp2[self.n_features-1,:]=w2[self.n_features-1,x[self.n_features-1],:,0]
        for i in range(self.n_features-2,-1,-1):
            tmp2[i,:]=cp.dot(w2[i,x[i],:,:],tmp2[i+1,:])
        tmp2[0,:]=cp.inner(w2[0,x[0],0,:],tmp2[1,:])
    
        #The derivative of each tensor is the contraction of the other tensors
        derivative[0,x[0],0,:]=2*cp.conj(tmp2[1,:])*mpscontracted
        derivative[self.n_features-1,
                   x[self.n_features-1],:,0]=2*cp.conj(tmp[self.n_features-2,:])*mpscontracted
        for i in range(1,self.n_features-1):
                derivative[i,x[i],:,:]=2*cp.conj(cp.outer(tmp[i-1,:],
                                            tmp2[i+1,:]))*mpscontracted

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """        
        
        w2=cp.reshape(self.w,(self.n_features,self.d,self.D,self.D))
        derivative=cp.zeros((self.n_features,self.d,self.D,self.D),dtype=cp.complex128)
        
        tmp=cp.zeros((self.n_features,self.D*self.D),dtype=cp.complex128)
        tmp2=cp.zeros((self.n_features,self.D*self.D),dtype=cp.complex128)
        tmp[0,:]=cp.tensordot(w2[0,:,0,:],cp.conj(w2[0,:,0,:]),axes=([0],[0])).reshape(self.D*self.D)
        for i in range(1,self.n_features-1):
            tmp[i,:]=cp.dot(tmp[i-1,:],cp.tensordot(w2[i,:,:,:],cp.conj(w2[i,:,:,:]),
                axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,self.D*self.D))
        tmp[self.n_features-1,:]=cp.inner(tmp[self.n_features-2,:],
                            cp.tensordot(w2[self.n_features-1,:,:,0],
                         cp.conj(w2[self.n_features-1,:,:,0]),
                         axes=([0],[0])).reshape(self.D*self.D))
        
        tmp2[self.n_features-1,:]=cp.tensordot(w2[self.n_features-1,:,:,0],
                cp.conj(w2[self.n_features-1,:,:,0]),
                axes=([0],[0])).reshape(self.D*self.D)
        for i in range(self.n_features-2,-1,-1):
            tmp2[i,:]=cp.dot(cp.tensordot(w2[i,:,:,:],cp.conj(w2[i,:,:,:]),
                axes=([0],[0])).transpose((0,2,1,3)).reshape(self.D*self.D,
                                    self.D*self.D),tmp2[i+1,:])
        tmp2[0,:]=cp.inner(cp.tensordot(w2[0,:,0,:],cp.conj(w2[0,:,0,:]),
                            axes=([0],[0])).reshape(self.D*self.D),tmp2[1,:])
        

        for j in range(self.d):
            derivative[0,j,0,:]=2*cp.dot(w2[0,j,0,:],
                                            tmp2[1,:].reshape(self.D,self.D))
            derivative[self.n_features-1,j,:,0]=2*cp.dot(w2[self.n_features-1,j,:,0],
                            tmp[self.n_features-2,:].reshape(self.D,self.D))
        for i in range(1,self.n_features-1):
            temp1=tmp[i-1,:].reshape(self.D,self.D)
            temp2=tmp2[i+1,:].reshape(self.D,self.D)
            
            for j in range(self.d):
                temp3=cp.dot(cp.dot(temp1.transpose(),w2[i,j,:,:]),temp2)
                derivative[i,j,:,:]=2*cp.copy(temp3)
                
        return derivative.reshape(self.m_parameters)


    def _weightinitialization(self, rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.w=cp.asarray(rng.normal(0, 1, self.m_parameters))\
                +1j*cp.asarray(rng.normal(0, 1, self.m_parameters))

    def _weightinitialization2(self,rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.m_parameters2=(self.n_features-2)*self.d*self.D*self.D+2*self.D*self.d
        return cp.asarray(rng.normal(0, 1, self.m_parameters2))\
                +1j*cp.asarray(rng.normal(0, 1, self.m_parameters2))
        
    def _likelihood_derivative(self, v):
        """Compute derivative of log-likelihood of configurations in v
        Parameters
        ----------
        v : numpy array, shape (n_samples,n_features)
            Configurations
        Returns
        -------
        update_w : numpy array, shape (m_parameters,)
            array of derivatives of the log-likelihood
        """
        update_w=cp.zeros(self.m_parameters,dtype=cp.complex128)
        for n in range(v.shape[0]):
            update_w -= self._logderivative(v[n,:])
        update_w += v.shape[0]*self._logderivativenorm()    
        update_w /= v.shape[0]
        return update_w