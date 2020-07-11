import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from displayData import *
from lrCostFunction import *
from sigmoid import *
from oneVsAll import *
from predictOneVsAll import *

# ===================== Part 1: Loading and Visualizing Data =====================

def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y

X, y = load_data('ex3data1.mat')
print(np.unique(y))  # 看下有几类标签

# dd.plot_an_image(X,y)
# dd.plot_100_image(X)

# ============ Part 2: Vectorize Logistic Regression ============

raw_X, raw_y = load_data('ex3data1.mat')
X = np.insert(raw_X, 0, 1, axis=1) # (5000, 401)
y = raw_y.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）
all_theta = one_vs_all(X, y, 1, 10)
# all_theta  # 每一行是一个分类器的一组参数

# ================ Part 3: Predict for One-Vs-All ================

y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))