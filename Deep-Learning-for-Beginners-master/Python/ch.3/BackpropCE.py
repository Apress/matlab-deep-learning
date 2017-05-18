import numpy as np
from Sigmoid import *


def BackpropCE(W1, W2, X, D):
    alpha = 0.9

    N = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Sigmoid(v)
        
        e     = d - y
        delta = e
        
        e1     = np.matmul(W2.T, delta)
        delta1 = y1*(1-y1) * e1
        
        dW1 = (alpha*delta1).reshape(4, 1) * x.reshape(1, 3)
        W1  = W1 + dW1
               
        dW2 = alpha * delta * y1
        W2  = W2 + dW2
    
    return W1, W2
