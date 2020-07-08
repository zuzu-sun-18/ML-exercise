import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from plotData import *
import costFunctionReg as cfr
import plotDecisionBoundary as pdb
import mapFeature as mf

data = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
plot_data(data)

# ===================== Part 1: Feature mapping =====================

X = mf.feature_map(data.values[:,:-1], 6)
y = data.values[:,-1]
theta = np.zeros(X.shape[1]) # 28 dim

# ===================== Part 2: Regularization and Accuracies =====================

import scipy.optimize as opt # 求最优 theta

def opt_minimize(theta, X, y, lamda=0):

    return opt.minimize(fun=cfr.cost_function_reg, x0=theta, args=(X, y, lamda), jac=cfr.gradient_reg, method='Newton-CG')

lamda = 0.2
result = opt_minimize(theta, X, y, lamda)
theta = result.x # 28 dim

# Plot boundary
# X×θ=0 (this is the line)

def plot_bound(theta, ax):

    data = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
    x_min = data.iloc[:,0].min()
    x_max = data.iloc[:,0].max()
    y_min = data.iloc[:,1].min()
    y_max = data.iloc[:,1].max()
    temp = np.array([(i,j) for i in np.linspace(x_min, x_max, 2000) for j in np.linspace(y_min, y_max, 2000)])
    data = mf.feature_map(temp, 6)
    temp = data[np.abs(data @ theta) < 0.0003]
    x = temp[:, 1]
    y = temp[:, 2]
    ax.scatter(x, y, label='desision bound')


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x=data[data['Accepted'] == 1]['Test 1'],
               y=data[data['Accepted'] == 1]['Test 2'], c='red', marker='o', label='y=1')
ax.scatter(x=data[data['Accepted'] == 0]['Test 1'],
               y=data[data['Accepted'] == 0]['Test 2'], c='blue', marker='x', label='y=0')
plot_bound(theta, ax)
ax.set_xlabel('Microchip Test 1')
ax.set_ylabel('Microchip Test 2')
ax.legend()
plt.show()

