import numpy as np
from Sigmoid import *
from Softmax import *
from TestMultiClass import *


def RealMultiClass():   
    W1, W2 = TestMultiClass()
    
    X = np.zeros((5, 5, 5))
    
    X[:, :, 0] = [[0,0,1,1,0],
                  [0,0,1,1,0],
                  [0,1,0,1,0],
                  [0,0,0,1,0],
                  [0,1,1,1,0]]
    
    X[:, :, 1] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [1,0,0,0,1],
                  [1,1,1,1,1]]
    
    X[:, :, 2] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [1,0,0,0,1],
                  [1,1,1,1,0]]
    
    X[:, :, 3] = [[0,1,1,1,0],
                  [0,1,0,0,0],
                  [0,1,1,1,0],
                  [0,0,0,1,0],
                  [0,1,1,1,0]]
    
    X[:, :, 4] = [[0,1,1,1,1],
                  [0,1,0,0,0],
                  [0,1,1,1,0],
                  [0,0,0,1,0],
                  [1,1,1,1,0]]
      
        
    N = 5
    for k in range(N):
        x  = np.reshape(X[:, :, k], (25, 1))
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Softmax(v)
        
        print("N = {}: ".format(k+1))
        print(y)

if __name__ == '__main__':
    RealMultiClass()