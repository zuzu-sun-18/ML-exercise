import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import *
import pandas as pd
import costFunction as cf
import plotDecisionBoundary as pdb
from sigmoid import *

data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1 score', 'Exam 2 score', 'Admission'])

# ===================== Part 1: Plotting =====================

plot_data(data)

# ===================== Part 2: Compute Cost and Gradient =====================

data.insert(0, 'ones', 1)
X = data.values[:,:-1]
y = data.values[:,-1]
theta = np.zeros(X.shape[1])
cost = cf.cost_function(theta, X, y)
grad = cf.gradient(theta, X, y)

# ===================== Part 3: Optimizing parameters theta using advanced algorithm =====================

import scipy.optimize as opt

result = opt.fmin_tnc(func=cf.cost_function, x0=theta, fprime=cf.gradient, args=(X, y))
theta = result[0]

# ===================== Part 4: Predict and Accuracies =====================

predict = sigmoid(np.dot(np.array([1, 45, 85]), theta))

def predict(theta, X):
    probability = sigmoid(X@theta)
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list

predictions = predict(theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(X)
print(accuracy)

# Plot boundary

theta = result[0]
pdb.plot_decision_boundary(theta, X, y)






