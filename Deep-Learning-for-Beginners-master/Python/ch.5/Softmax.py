import numpy as np


def Softmax(x):
    x  = np.subtract(x, np.max(x))        # prevent overflow
    ex = np.exp(x)
    
    return ex / np.sum(ex)