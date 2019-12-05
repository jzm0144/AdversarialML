import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def Get_Casis_CUDataset():
    X = []
    Y = []
    with open("CASIS-25_CU.txt", "r") as feature_file:
        for line in feature_file:
            line = line.strip().split(",")
            Y.append(line[0][:4])
            X.append([float(x) for x in line[1:]])
    return np.array(X), np.array(Y)
      
class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()