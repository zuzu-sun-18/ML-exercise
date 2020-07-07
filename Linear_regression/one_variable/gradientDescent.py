
import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):

    J_history = []

    for i in range(num_iters):

        grad0 = np.mean(np.dot(X, theta) - y)
        grad1 = np.mean((np.dot(X, theta) - y) * (X[:, 1].reshape([len(X), 1])))
        theta[0] = theta[0] - alpha * grad0
        theta[1] = theta[1] - alpha * grad1

        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


