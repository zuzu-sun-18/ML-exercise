import numpy as np

def normal_eqn(X, y):

    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta