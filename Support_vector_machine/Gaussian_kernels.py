from Linear_kernal import *

# ===================== Part 5: Training SVM with RBF Kernel (Dataset 2) =====================
'''
def gaussKernel(x1, x2, sigma):
    return np.exp(- ((x1 - x2) ** 2).sum() / (2 * sigma ** 2))

print(gaussKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))  # 0.32465246735834974
'''
path = 'ex6data2.mat'
mat = loadmat(path)
X2 = mat['X']
y2 = mat['y']
#plotData(X2, y2)

# 用高斯核函数拟合模型
sigma = 0.1
gamma = np.power(sigma,-2.)/2
clf = svm.SVC(C=1, kernel='rbf', gamma=gamma)
modle = clf.fit(X2, y2.flatten())
# 决策边界
plotData(X2, y2)
plotBoundary(modle, X2)

# ===================== Part 6: Visualizing Dataset 3 =====================

path = 'ex6data3.mat'
mat3 = loadmat(path)
X3, y3 = mat3['X'], mat3['y']
Xval, yval = mat3['Xval'], mat3['yval']
#plotData(X3, y3)

Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for C in Cvalues:
    for sigma in sigmavalues:
        gamma = np.power(sigma,-2.)/2
        model = svm.SVC(C=C,kernel='rbf',gamma=gamma)
        model.fit(X3, y3.flatten())
        this_score = model.score(Xval, yval) #score（X，y）：返回给定测试集合对应标签的平均准确率
        if this_score > best_score:
            best_score = this_score
            best_pair = (C, sigma)
print('best_pair={}, best_score={}'.format(best_pair, best_score))

model = svm.SVC(C=1., kernel='rbf', gamma = np.power(.1, -2.)/2)
model.fit(X3, y3.flatten())
plotData(X3, y3)
plotBoundary(model, X3)


