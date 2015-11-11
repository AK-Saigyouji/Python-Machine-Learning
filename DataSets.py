"""
Functions for creating data sets for the testing of algorithms.
"""

import numpy as np
from scipy.stats import norm

def get_binary_data(rows, cols, scale = 50, add_noise = False, noise_var = 1.0):
    """ Generate a dense matrix X with the given number of rows and columns, and
    a vector of binary labels Y that depend linearly on the features.
    
    Args:
        rows (int): the number of observations in the data set
        cols (int): the number of features in the data set
        add_noise (bool): whether to include noise (random error) in the
            labels with mean 0 and variance noise_var (default: False)
        noise_var (float): the variance of the noise, if included (default = 1.0)
    Returns:
        Y (array): a 1D array of binary labels for the data
        X (array): a rows by cols ndarray of random i.i.d. data
        weights (array): the weights applied to the columns of X
            to generate Y
    """
    
    Z, X, weights = get_dense_data(rows = rows, cols = cols, 
                                   scale = scale, add_noise = add_noise, 
                                   noise_var = noise_var)
    Y = (Z >= 0) * 1
    return Y, X, weights

def get_dense_data(rows, cols, scale = 50, add_noise = False, noise_var = 1.0):
    """Generate a dense matrix X with the given number of rows and columns, and
    a random linear combination Y of the features.
    
    Args:
        rows (int): the number of observations in the data set
        cols (int): the number of features in the data set
        scale (float): variance for the weights
        add_noise (bool): whether to include noise (random error) in the
            labels with mean 0 and variance noise_var (default: False)
        noise_var (float): the variance of the noise, if included
    Returns:
        Y (array): a 1D array of labels for the data
        X (array): a rows by cols ndarray of random i.i.d. data
        weights (array): the weights applied to the columns of X
            to generate Y
    """
    
    X = np.array([norm.rvs(size = rows) for n in range(cols)]).transpose()
    weights = norm.rvs(size = cols, scale = scale)
    noise = add_noise * norm.rvs(size = rows, scale = noise_var)
    Y = np.sum(X * weights, axis = 1) + noise
    return Y, X, weights