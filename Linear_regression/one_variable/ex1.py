import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *

# ===================== Part 1: Plotting =====================

data = np.loadtxt('ex1data1.txt', delimiter=',') # 指定分隔符
X = data[:, 0]
y = data[:, 1]
y = y.reshape([len(y), 1]) # 不reshape一下会报错
m = y.size
plt.ion()
plt.figure(0)
plot_data(X, y)
input('Program paused. Press enter to continue.\n') # 调用input函数以达到暂停的目的

# ===================== Part 2: Gradient descent =====================

X = np.c_[np.ones(m), X]  # 数组拼接
print(X.shape)
theta = np.zeros((2,1))  # initialize fitting parameters
print(theta.shape)
iterations = 1500
alpha = 0.01
# Compute and display initial cost
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(theta.reshape(2)))

# Plot the linear fit
plt.figure(0)
plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend()
plt.ioff()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print(predict1)
predict2 = np.dot(np.array([1, 7]), theta)
print(predict2)

# ===================== Part 3: Visualizing J(theta0, theta1) =====================

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(ys, xs, J_vals, cmap='rainbow')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()

plt.figure()
lvls = np.logspace(-2, 3, 20)
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.plot(theta[0], theta[1], c='r', marker="x")
print(theta[0], theta[1])
plt.show()
