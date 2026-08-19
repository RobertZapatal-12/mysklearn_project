import numpy as np

def mean_absolute_error(y_pred, y_true):
    n = len(y_true)
    mae = (1/n) * np.sum(np.absolute(y_pred - y_true))
    return mae

def mean_squared_error(y_pred, y_true):
    n = len(y_true)
    mse = (1/n) * np.sum(np.square(y_pred - y_true))
    return mse

def root_mean_squared_error(y_pred, y_true):
    n = len(y_true)
    rmse = np.sqrt((1/n) * np.sum(np.square(y_pred - y_true)))
    return rmse