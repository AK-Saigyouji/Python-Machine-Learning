"""
An implementation of linear regression using gradient descent with mean squared
error as the cost function being optimized. Uses a constant learning rate.
"""

from __future__ import division
import numpy as np 

__all__ = ['gradient_descent', 'get_predictions']

def gradient_descent(X, Y, learning_rate = 1.0, iterations = 1500, 
                     regularization = 0):
    """ Perform linear regression with gradient descent. Use mean squared error 
    as the cost function, a constant learning rate, and a fixed number of 
    iterations.
    
    Args:
        X (ndarray): a 2D array of features for training data, where each row 
            is an obsevation and the columns are features. 
        Y (array): an array of known values corresponding to each row in X.
        learning_rate (float): the proportion of the gradient update to use 
            when updating weights (default: 1.0).
        iterations (int): the number of updates to apply (default: 1500).
        regularization (float): what proportion of the L2 norm of the weights
            to include in the cost function (default: 0).
    Returns: 
        weights (array): the final coefficients after running gradient descent.
        errors (array): the mean squared errors after each iteration.
    """
    
    error_history = np.empty(iterations)
    X_norm, mean, std = normalize(X)
    X_norm = insert_ones(X_norm)
    weights = initialize_weights(X_norm)
    for n in range(iterations):
        error_history[n] = MSE(weights, X_norm, Y, regularization)
        weights = gradient_update(weights, X_norm, Y, learning_rate, 
                                  regularization)
    weights = denormalize_weights(weights, mean, std)
    return weights, error_history

def gradient_update(weights, X, Y, learning_rate, regularization):
    """ Update the weights according to gradient descent."""    
    alpha = learning_rate
    m = len(Y)
    return weights - (alpha / m) * gradient(weights, X, Y, regularization)  

def gradient(weights, X, Y, regularization):
    """ Return the gradient of the cost function."""
    regularization_term = regularization * sum(weights[1:])
    return X.transpose().dot(residual(weights, X, Y)) + regularization_term
    
def MSE(weights, X, Y, regularization):
    """ Compute the mean squared error."""
    base_penalty = sum((residual(weights, X, Y)**2) / len(Y)) 
    regularization_penalty = regularization * sum(weights[1:]**2)
    return base_penalty + regularization_penalty   

def residual(weights, X, Y):
    """ Compute the residual."""
    return X.dot(weights) - Y

def normalize(X):
    """ Normalize X so that it has mean 0 and std 1."""    
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    X_norm = (X - mean) / std 
    return X_norm, mean, std
    
def denormalize_weights(weights, mean, std):
    """ Undo the effects of normalization on the weights."""    
    weights[0] -= sum(weights[1:] * mean / std)
    weights[1:] = weights[1:] / std
    return weights

def insert_ones(X):
    """ Insert a column of ones into the left side of X."""    
    # This step allows X to be a one-dimensional array. 
    X = array_to_ndarray(X)
    num_rows = X.shape[0]
    return np.hstack((np.ones((num_rows, 1)), X))
    
def array_to_ndarray(X):
    """ Return a multidimensional version of X if it isn't already."""    
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    return X
    
def initialize_weights(X):
    """ Return an array of weights to begin gradient descent."""    
    num_columns = X.shape[1]
    return np.zeros(num_columns)
    
def get_predictions(new_data, weights):
    """ Use weights from linear regression to make predictions on new data."""
    return insert_ones(new_data).dot(weights)