import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def feature_map(data, power=1):
    poly = PolynomialFeatures(power)
    X = poly.fit_transform(data)
    return X

