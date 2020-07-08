import matplotlib.pyplot as plt
import numpy as np

def plot_data(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=data[data['Accepted'] == 1]['Test 1'],
               y=data[data['Accepted'] == 1]['Test 2'], c='red', marker='o', label='y=1')
    ax.scatter(x=data[data['Accepted'] == 0]['Test 1'],
               y=data[data['Accepted'] == 0]['Test 2'], c='blue', marker='x', label='y=0')
    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')
    ax.legend()
    plt.show()
