import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_normalize import *
from gradientDescent import *
from normal_eqn import *

# ===================== Part 1: Feature Normalization =====================

# When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge
# much more quickly.

data = pd.read_csv('ex1data2.txt', sep=',', header=None, names=['size', 'bedrooms', 'price'])
# <class 'pandas.core.frame.DataFrame'>
y = data.values[:,-1] # <class 'numpy.ndarray'>
y = y.reshape((-1,1))
X = data.iloc[:,:-1] # <class 'pandas.core.frame.DataFrame'>

#X = np.c_[np.ones(len(y)), X]
#theta = normal_eqn(X, y)
#input("bbbbb")

X_norm = feature_normalize(X) # <class 'pandas.core.frame.DataFrame'>
X_norm = np.c_[np.ones(len(y)), X_norm]  # <class 'numpy.ndarray'>

# ===================== Part 2: Gradient Descent =====================

num_iters = 200
alpha = [1, 0.3, 0.1, 0.03, 0.01]

theta,ax = plt.subplots(figsize=(10,6))
for i in alpha:
    theta = np.zeros(3)
    theta = theta.reshape(3,1)
    theta, J_history = gradient_descent_multi(X_norm, y, theta, i, num_iters)
    ax.plot(J_history, label='lr=%.2f' % (i))

ax.set_xlabel('iterations')
ax.set_ylabel('Cost')
ax.legend()
plt.show()


