import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lamda=0):

    cost = np.mean(- y * np.log(sigmoid(np.dot(X, theta))) - (1 - y) * np.log(1 - sigmoid(np.dot(X, theta)))) + lamda / (2 * X.shape[0]) * sum(theta[1:] * theta[1:])
    # 没有theta[0]

    return cost

def gradient_reg(theta, X, y, lamda=0):

    temp = np.zeros(X.shape[1])
    temp[1:] = lamda / X.shape[0] * theta[1:] # 不惩罚第一项 theta[0]
    grad = (1/X.shape[0]) * np.dot(X.T, sigmoid(np.dot(X, theta)) - y) + temp

    return grad
