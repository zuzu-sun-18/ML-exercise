import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn import svm
import re # 电子邮件处理的正则表达式

# 一个英文分词算法(Poter stemmer)
import nltk, nltk.stem.porter

with open('emailSample1.txt', 'r') as f:
    email = f.read()

def process_email(email):
    """做除了Word Stemming和Removal of non-words的所有处理"""
    email = email.lower()
    email = re.sub('<[^<>]>', ' ', email)
    # 匹配<开头，然后所有不是< ,> 的内容，知道>结尾，相当于匹配<...>
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email )
    # 匹配//后面不是空白字符的内容，遇到空白字符则停止
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email


def email2TokenList(email):
    """预处理数据，返回一个干净的单词列表"""

    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()

    email = process_email(email)

    # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # 遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token);
        # Use the Porter stemmer to 提取词根
        stemmed = stemmer.stem(token)
        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue
        tokenlist.append(stemmed)

    return tokenlist

def email2VocabIndices(email, vocab):
    """提取存在单词的索引"""
    token = email2TokenList(email)
    index = [i for i in range(len(vocab)) if vocab[i] in token]
    return index

def email_feature_vector(email):
    '''将email的单词转换为特征向量0/1'''
    df = pd.read_table('vocab.txt', names=['words'])
    vocab = df.values # Datafram转换为ndarray
    vector = np.zeros(len(vocab))
    vecab_indices = email2VocabIndices(email, vocab)
    for i in vecab_indices:
        vector[i] = 1
    return vector

vector = email_feature_vector(email)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))

# Training SVM for Spam Classification
mat1 = loadmat('spamTrain.mat')
X, y = mat1['X'], mat1['y'].flatten()
# Test set
mat2 = loadmat('spamTest.mat')
Xtest, ytest = mat2['Xtest'], mat2['ytest']

clf = svm.SVC(C=0.1, kernel='linear')
clf.fit(X, y)

#predTrain = clf.score(X, y)
#predTest = clf.score(Xtest, ytest)
# print(predTrain, predTest)

# 预测自己的email是否是spam

with open('own_email_test.txt', 'r') as f:
    email_test = f.read()
X_vector = email_feature_vector(email_test)
print('length of vector = {}\nnum of non-zero = {}'.format(len(X_vector), int(X_vector.sum())))
X_vector = X_vector.reshape(1,1899)
result = clf.predict(X_vector)
print(result)