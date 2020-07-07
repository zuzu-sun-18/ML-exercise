import matplotlib.pyplot as plt

def plot_data(x, y):

    plt.scatter(x, y, c='red', marker='x', label='train_data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    #plt.show()