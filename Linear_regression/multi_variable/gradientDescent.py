import numpy as np
from computeCost import *


def gradient_descent_multi(X, y, theta, alpha, num_iters):

    J_history = []
    J_history.append(compute_cost(X, y, theta))
    grad = np.zeros(len(theta))

    for i in range(0, num_iters):
        for j in range(len(theta)):
            grad[j] = np.mean((np.dot(X, theta) - y) * (X[:, j].reshape([len(X), 1])))
        for k in range(len(theta)):
            theta[k] = theta[k] - alpha * grad[k]

        J_history.append(compute_cost(X, y, theta))

    return theta, J_history