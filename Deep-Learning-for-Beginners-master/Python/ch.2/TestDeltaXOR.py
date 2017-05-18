import numpy as np
from Sigmoid import *
from DeltaXOR import *


def TestDeltaXOR():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    W = 2*np.random.random((1, 3)) - 1
    
    for _epoch in range(40000):     #train
        W = DeltaXOR(W, X, D)
        
    N = 4                           #inference
    for k in range(N):              
        x = X[k,:].T
        v = np.matmul(W, x)
        y = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    TestDeltaXOR()