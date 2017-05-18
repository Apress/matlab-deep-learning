import numpy as np
from Sigmoid import *
from Softmax import *


def MultiClass(W1, W2, X, D):
    alpha = 0.9
    
    N = 5
    for k in range(N):
        x = np.reshape(X[:,:,k], (25, 1))
        d = D[k, :].T
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Softmax(v)
            
        e     = d - y
        delta = e
        
        e1     = np.matmul(W2.T, delta)
        delta1 = y1*(1-y1) * e1
        
        dW1 = alpha * delta1 * x.T
        W1  = W1 + dW1
        
        dW2 = alpha * delta * y1.T
        W2  = W2 + dW2
        
    return W1, W2