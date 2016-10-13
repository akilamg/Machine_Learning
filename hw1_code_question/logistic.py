""" Methods for doing logistic regression."""

import numpy as np
import math as math
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    new_data = np.ones((data.shape[0], weights.shape[0]))
    new_data[:, :-1] = data
    z = new_data.dot(weights)
    y = sigmoid( z )

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = np.sum( -targets.T.dot( np.log(y) ) )
    frac_correct = 0
    for t,y_ in zip(targets,y):
        if t == round(y_):
            frac_correct += 1
    frac_correct = frac_correct/float(y.shape[0])

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        new_data = np.ones((data.shape[0], data.shape[1] + 1))
        new_data[:, :-1] = data
        z = new_data.dot(weights)
        f = np.squeeze( ( np.ones((targets.shape) ) - targets).T.dot( z ) )+ np.sum( np.log( ( 1.0 + np.exp( -z ) ) ) )
        df = new_data.T.dot( y - targets )

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """
    y = logistic_predict(weights, data)

    # TODO: Finish this function
    new_data = np.ones((data.shape[0], data.shape[1] + 1))
    new_data[:, :-1] = data

    bias_weights = weights[weights.shape[0]-1]
    non_bias_weights = weights[:-1]

    I = np.ones(non_bias_weights.shape)
    hyper_field = hyperparameters['weight_decay'] * I
    z = new_data.dot(weights)
    non_bias_weights_pow_2 = np.square(non_bias_weights)

    log_p_w = -np.squeeze(hyper_field.T.dot( non_bias_weights_pow_2 ))/2.0  - ( np.log( 2.0 * math.pi ) - np.log( hyperparameters['weight_decay'] ) ) / 2.0
    f = np.squeeze( ( np.ones((targets.shape) ) - targets ).T.dot(z)) + np.sum(np.log((1.0 + np.exp(-z)))) + log_p_w
    df_0 = np.sum(y - targets).reshape(1,1)
    df_j = data.T.dot(y - targets) - hyper_field.T.dot(non_bias_weights)
    df = np.concatenate((df_j, df_0), axis=0)

    return f, df
