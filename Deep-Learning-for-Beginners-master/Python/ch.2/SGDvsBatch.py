from DeltaSGD import *
from DeltaBatch import *
from Sigmoid import *
import matplotlib.pyplot as plt


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

W1 = 2*np.random.random((1, 3)) - 1
W2 = np.array(W1)

for epoch in range(1000):
    W1 = DeltaSGD(W1, X, D)
    W2 = DeltaBatch(W2, X, D)
    
    es1 = 0
    es2 = 0
    N   = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        es1 = es1 + (d - y1)**2
        
        v2 = np.matmul(W2, x)
        y2 = Sigmoid(v2)
        es2 = es2 + (d - y2)**2
        
    E1[epoch] = es1/N
    E2[epoch] = es2/N
    
    
SGD,   = plt.plot(E1, 'r')
Batch, = plt.plot(E2, 'b:')
plt.xlabel("Epoch")
plt.ylabel("Average of Training Error")
plt.legend([SGD, Batch], ['SGD', 'Batch'])
plt.show()