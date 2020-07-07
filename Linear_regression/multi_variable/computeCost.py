import numpy as np

def compute_cost(X, y, theta):

    cost = 0.5 * np.mean(np.square(np.dot(X, theta) - y))

    return cost