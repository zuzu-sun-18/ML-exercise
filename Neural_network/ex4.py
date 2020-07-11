import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告
from sklearn.preprocessing import OneHotEncoder

# ===================== Part 1: Loading and Visualizing Data =====================

def load_mat(path):
    data = loadmat('ex4data1.mat')  # return a dict
    X = data['X']
    y = data['y'].flatten()
    return X, y

def plot_100_images(X):
    """随机画100个数字"""
    index = np.random.choice(range(5000), 100)
    images = X[index]
    fig, ax_array = plt.subplots(10, 10, sharey=True, sharex=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(images[r*10 + c].reshape(20,20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

X,y = load_mat('ex4data1.mat')
#(X.shape,type(X),y.shape,type(y)) (5000, 400) <class 'numpy.ndarray'> (5000,) <class 'numpy.ndarray'>
#plot_100_images(X)

# ===================== Part 2: Loading Parameters =====================

def expand_y(y): # 把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1

    result = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i-1] = 1
        result.append(y_array)

    return np.array(result)

raw_X, raw_y = load_mat('ex4data1.mat')
X = np.insert(raw_X, 0, 1, axis=1)
y = expand_y(raw_y)
#(X.shape,type(X),y.shape,type(y))  (5000, 401) <class 'numpy.ndarray'> (5000, 10) <class 'numpy.ndarray'>

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']
t1, t2 = load_weight('ex4weights.mat')
# (t1.shape, t2.shape) (25, 401) (10, 26)

def serialize(a, b):
    '''展开参数'''
    return np.r_[a.flatten(),b.flatten()]
theta = serialize(t1, t2)  # 扁平化参数，25*401+10*26=10285
# theta.shape   (10285,)

def deserialize(seq):
    '''提取参数'''
    return seq[:25*401].reshape(25, 401), seq[25*401:].reshape(10, 26)

# ===================== Part 3: Compute Cost (Feedforward) =====================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feed_forward(theta, X, ):
    '''得到每层的输入和输出'''
    t1, t2 = deserialize(theta)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = X # array
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ t2.T
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3

a1, z2, a2, z3, h = feed_forward(theta, X)

# （不带正则化项）
def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    return J
# print(cost(theta, X, y)) 0.2876291651613189

# （带正则化项）注意不要将每层的偏置项正则化
def regularized_cost(theta, X, y, l=1):
    '''正则化时忽略每层的偏置项，也就是参数矩阵的第一列'''
    t1, t2 = deserialize(theta) # array
    reg = np.sum(t1[:,1:] ** 2) + np.sum(t2[:,1:] ** 2)  # or use np.power(a, 2)
    return l / (2 * len(X)) * reg + cost(theta, X, y)
#print(regularized_cost(theta, X, y, 1)) #0.38376985909092354

# ===================== Part 4: Backpropagation =====================

def sigmoid_gradient(z):  #Sigmoid gradient S函数导数
    return sigmoid(z) * (1 - sigmoid(z))

def random_init(size):   #Random initialization 随机初始化
    '''从服从的均匀分布的范围中随机返回size大小的值
    可以选择 e = 0.12 这个范围的值来确保参数足够小'''
    return np.random.uniform(-0.12, 0.12, size)


def gradient(theta, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y  # (5000, 10)
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = d3.T @ a2  # (10, 26)
    D1 = d2.T @ a1  # (25, 401)
    D = (1 / len(X)) * serialize(D1, D2)  # (10285,)

    return D

def regularized_gradient(theta, X, y, l=1):
    """不惩罚偏置单元的参数"""
    a1, z2, a2, z3, h = feed_forward(theta, X)
    D1, D2 = deserialize(gradient(theta, X, y))
    t1[:, 0] = 0
    t2[:, 0] = 0
    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2

    return serialize(reg_D1, reg_D2)

# ===================== Part 5: Gradient checking =====================

def gradient_checking(theta, X, y, e):  #这个运行很慢，谨慎运行

    def a_numeric_grad(plus, minus):
        """
        对每个参数theta_i计算数值梯度，即理论梯度。
        """
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (e * 2)

    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy()  # deep copy otherwise you will change the raw theta
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_gradient(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)

    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(
            diff))

# ===================== Part 5: Learning parameters using fmincg 优化参数 =====================

def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res

res = nn_training(X, y)#慢
#res

def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))
