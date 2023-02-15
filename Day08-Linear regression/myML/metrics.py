import numpy as np
def accuracy(y_true, y_pred):
    return sum(y_pred == y_true) / len(y_pred)

def mean_square_error(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)/len(y_true)

def root_mean_square_error(y_true, y_pred):
    return (np.sum((y_true-y_pred)**2)/len(y_true))**0.5

def mean_absolute_error(y_true, y_pred):
    return np.sum(abs(y_true-y_pred))/len(y_true)

def r2(y_true, y_pred):
    return 1-mean_square_error(y_true, y_pred)/np.var(y_true)