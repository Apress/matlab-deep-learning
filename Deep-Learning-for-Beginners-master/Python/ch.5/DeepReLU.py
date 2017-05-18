import numpy as np
from Softmax import *
from ReLU import *


def DeepReLU(W1, W2, W3, W4, X, D):
    alpha = 0.01
    
    N = 5
    for k in range(N):
        x  = np.reshape(X[:, :, k], (25, 1))
        
        v1 = np.matmul(W1, x)
        y1 = ReLU(v1)
        
        v2 = np.matmul(W2, y1)
        y2 = ReLU(v2)
        
        v3 = np.matmul(W3, y2)
        y3 = ReLU(v3)
        
        v = np.matmul(W4, y3)
        y = Softmax(v)
        
        d     = D[k, :].T 
        e     = d - y
        delta = e
        
        e3     = np.matmul(W4.T, delta)
        delta3 = (v3 > 0) * e3
        
        e2     = np.matmul(W3.T, delta3)
        delta2 = (v2 > 0) * e2
        
        e1     = np.matmul(W2.T, delta2)
        delta1 = (v1 > 0) * e1
        
        dW4 = alpha * delta * y3.T
        W4  = W4 + dW4
        
        dW3 = alpha * delta3 * y2.T
        W3  = W3 + dW3
        
        dW2 = alpha * delta2 * y1.T
        W2  = W2 + dW2
        
        dW1 = alpha * delta1 * x.T
        W1  = W1 + dW1
    
    return W1, W2, W3, W4
    