import numpy as np

def mean_absolute_error(y_pred, y_true):
    """Calculates the mean absoluted error of the predicted labels.
    
    Args:
        y_pred (np.ndarray): Predicted values.
        y_true(pd.Series, np.ndarrray): True labels
    
    Returns:
        The value of the Mean Absolute Error
    """
    mae = np.mean(np.absolute(y_pred - y_true))
    return mae

def mean_squared_error(y_pred, y_true):
    """Calculates the mean squared error of the predicted labels.
    
    Args:
        y_pred (np.ndarray): Predicted values.
        y_true(pd.Series, np.ndarrray): True labels
    
    Returns:
        The value of the Mean Squared Error
    """
    mse = np.mean(np.square(y_pred - y_true))
    return mse

def root_mean_squared_error(y_pred, y_true):
    """Calculates the root mean squared error of the predicted labels.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_true(pd.Series, np.ndarrray): True labels

    Returns:
        The value of the Root Mean Squared Error
    """
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    return rmse


def r2_score(y_pred, y_true):
    """Calculates the r2 score of the predicted labels.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_true(pd.Series, np.ndarrray): True labels

    Returns:
        The value of the R2 Score
    """
    y_true_avg = np.mean(y_true)
    ss_num = np.sum(np.square(y_true - y_pred))
    ss_den = np.sum(np.square(y_true - y_true_avg))
    r2_score = 1 - (ss_num / ss_den)
    return r2_score


