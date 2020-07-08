import numpy as np
from sigmoid import *


def cost_function(theta, X, y):

    cost = np.mean(- y * np.log(sigmoid(np.dot(X, theta))) - (1 - y) * np.log(1 - sigmoid(np.dot(X, theta))))
    return cost

def gradient(theta, X, y):

    grad = (1/X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)

    return grad