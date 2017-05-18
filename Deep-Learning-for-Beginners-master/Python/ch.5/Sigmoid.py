import numpy as np


def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))