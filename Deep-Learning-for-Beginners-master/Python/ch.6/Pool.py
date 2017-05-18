import numpy as np
from scipy import signal

    
def Pool(x):
    (xrow, xcol, numFilters) = x.shape
    y = np.zeros((int(xrow/2), int(xcol/2), numFilters))
    
    for k in range(numFilters):
        filter = np.ones((2,2)) / (2*2)
        image  = signal.convolve2d(x[:, :, k], filter, 'valid')
        
        y[:, :, k] = image[::2, ::2]

    return y
