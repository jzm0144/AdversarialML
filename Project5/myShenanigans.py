import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import ipdb as ipdb
from random import randint



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

def readForPertubration():
    data = []
    with open("CASIS-25_CU.txt", "r") as feature_file:
        for line in feature_file:
            line = line.strip().split(",")
            thisY = line[0][:4]
            thisX = [float(x) for x in line[1:]]
            thisX.append(thisY)
            data.append(thisX)
    return data

def perturb(vector, copies = 24, numPerturbations = 5):
    x = vector[:-1]
    y = vector[-1]

    df = pd.read_csv('Mask.csv')    
    mask = df['mask'].values

    offPositions = []
    for index in range(len(mask)):
        if mask[index] == 0:
            offPositions.append(index)
    # Apply the Mask
    x = np.multiply(np.array(x), mask)
    x_buf = x
    # First row is the actual data example
    pX = np.zeros((copies+1, len(x)))
    pX[0,:] = x
    pY = []
    pY.append(y)


    for turn in range(copies):
        # Generate indices
        indices = np.random.randint(95, size=numPerturbations)
        R = np.random.rand(numPerturbations)

        for q in range(len(indices)):
            index = indices[q]
            temp = True

            for i in offPositions:
                if i == index:
                    temp = False

            if temp == True:
                # Update Rules
                #x[index] = R[q]
                x[index] = R[q] * x_buf[index]

        pX[turn+1,:] = x
        pY.append(y)


    return pX, pY

def create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt'):
    data = readForPertubration()

    ###################################
    xData= np.zeros((25*4*25,95))
    yData= []
    for i in range(len(data)):
        print(i)
        pX, pY = perturb(data[i],copies=24)
        xData[25*i:25*(i+1), :] = pX
        for item in pY:
            yData.append(item)
    
    print('******************')
    print("Perturbation Done:")
    return xData, np.array(yData)














