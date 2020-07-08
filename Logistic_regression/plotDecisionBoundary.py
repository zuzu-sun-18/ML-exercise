import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotData import *
from mapFeature import *


def plot_decision_boundary(theta, X, y):

# X×θ=0 (this is the line)

# θ0+x1*θ1+x2*θ2=0

    x = [X[:, 1].min(), X[:, 1].max()]
    y = [-(theta[0] + theta[1] * x[0]) / theta[2],
         -(theta[0] + theta[1] * x[1]) / theta[2]]
    fig, ax = plt.subplots(figsize=(10, 6))
    data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1 score', 'Exam 2 score', 'Admission'])
    data.insert(0, 'ones', 1)

    data[data['Admission'] == 0].plot(x='Exam 1 score', y='Exam 2 score', kind='scatter', c='red', marker='o', ax=ax,
                                        label='Not admitted')
    data[data['Admission'] == 1].plot(x='Exam 1 score', y='Exam 2 score', kind='scatter', c='blue', marker='x', ax=ax,
                                        label='Admitted')
    ax.plot(x, y)
    plt.show()



