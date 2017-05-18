import numpy as np


def Dropout(y, ratio):
    ym = np.zeros_like(y)
    
    num = round(y.size*(1-ratio))
    idx = np.random.choice(y.size, num, replace=False)
    ym[idx] = 1.0 / (1.0 - ratio)
    
    return ym