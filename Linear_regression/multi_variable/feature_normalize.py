import numpy as np
import pandas as pd

def feature_normalize(X):

    X_norm = (X - X.mean()) / X.std()

    return X_norm