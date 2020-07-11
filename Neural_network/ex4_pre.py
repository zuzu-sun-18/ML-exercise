import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('ex3weights.mat')
print(theta1.shape, theta2.shape)

def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

X, y = load_data('ex3data1.mat')
y = y.flatten()
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

print(X.shape, y.shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

a1 = X
z2 = np.dot(a1,theta1.T)
print(z2.shape)
z2 = np.insert(z2, 0, 1, axis=1)
print(z2.shape)
a2 = sigmoid(z2)
print(a2.shape)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1) + 1
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))  # accuracy = 97.52%