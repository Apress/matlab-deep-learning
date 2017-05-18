import numpy as np
import matplotlib.pyplot as plt
from Sigmoid import *
from BackpropCE import *
from BackpropXOR import *


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0],
              [0],
              [1],
              [1]])

E1 = np.zeros(1000)
E2 = np.zeros(1000)

W11 = 2*np.random.random((4, 3)) - 1
W12 = 2*np.random.random((1, 4)) - 1
W21 = np.array(W11)
W22 = np.array(W12)

for _epoch in range(1000):
    W11, W12 = BackpropCE(W11, W12, X, D)
    W21, W22 = BackpropXOR(W21, W22, X, D)

    es1 = 0
    es2 = 0 
    N   = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]

        v1  = np.matmul(W11, x)
        y1  = Sigmoid(v1)
        v   = np.matmul(W12, y1)
        y   = Sigmoid(v)
        es1 = es1 + (d - y)**2
        
        v1  = np.matmul(W21, x)
        y1  = Sigmoid(v1)
        v   = np.matmul(W22, y1)
        y   = Sigmoid(v)
        es2 = es2 + (d - y)**2
        
    E1[_epoch] = es1 / N
    E2[_epoch] = es2 / N
    

CE,  = plt.plot(E1, 'r')
SSE, = plt.plot(E2, 'b:')
plt.xlabel('Epoch')
plt.ylabel('Average of Training error')
plt.legend([CE, SSE], ["Cross Entropy", "Sum of Squared Error"])
plt.show()