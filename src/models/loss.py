import numpy as np

def p1(state, target):
    return np.abs((state[:,:,0].astype(np.int64)-target[:,:,0].astype(np.int64))).sum()

def p2(state, target):
    return np.sum((state-target)**2)

