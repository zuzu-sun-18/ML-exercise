import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

# ===================== Part 1: Loading and Visualizing Data =====================

path = 'ex6data1.mat'
mat = loadmat(path)
# print(mat.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
X = mat['X']
y = mat['y']

def plotData(X, y):
    plt.figure(figsize=(8,5))
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    #plt.show()
plotData(X, y)

# ===================== Part 2: Training Linear SVM =====================

# np.meshgrid()生成网格点，再对每个网格点进行预测，最后画出等高线图，即决策边界。
def plotBoundary(clf, X):
    '''plot decision bondary'''
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    y_min, y_max = X[:,1].min()*1.1,X[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500)) # 画网格点
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 用训练好的分类器对网格点进行预测
    Z = Z.reshape(xx.shape) # 转换成对应的网格点
    plt.contour(xx, yy, Z) # 等高线图，画出0/1分界线

models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
clfs = [model.fit(X, y.ravel()) for model in models]

title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
for model,title in zip(clfs,title):
    plt.figure(figsize=(8,5))
    plotData(X, y)
    plotBoundary(model, X)
    plt.title(title)
    #plt.show()

