import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from pickle import dump, load
import ipdb as ipdb
import pandas as pd
from random import randint


import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ask_the_ensemble(input_name = "AdversarialText.txt"):

    X, Y = Get_Casis_CUDataset(input_name)

    # Train the Mode, the models are already trained
    # Train()

    X = preprocessVector(X)

    yHat = getPredictions(X)
    return yHat


def Get_Casis_CUDataset(filename = "CASIS-25_CU.txt"):
    X = []
    Y = []
    with open(filename, "r") as feature_file:
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



def Train():
    CU_X, Y = create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt')

    fold_accuracy = []

    for repeat in range(1): # it was 10 intitially
        #-----------------------------Classifiers------------------------
        # SVM with Radial Basis Function
        rbfsvm = svm.SVC()
        # Linear SVM
        lsvm = svm.LinearSVC()
        # Multilayer Perceptron
        mlp = MLPClassifier(hidden_layer_sizes = (95,25),
                            activation = ('relu'),
                            max_iter=1000)
        # Decision Tree
        dTree = DecisionTreeClassifier(random_state=0)
        # Random Forests
        RF = RandomForestClassifier(random_state=0)
        # KNN, replaced my NB with KNN later. Naive Bayes
        KNN = KNeighborsClassifier(n_neighbors=3)

        # Data Manipulation, Preprocessing, Training and Testing

        # 4-Fold CrossValidation with Shuffling
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        

        scaler = StandardScaler()
        tfidf = TfidfTransformer(norm=None)
        dense = DenseTransformer()

        for train, test in skf.split(CU_X, Y):
            #train split
            CU_train_data = CU_X[train]
            train_labels = Y[train]
            
            #test split
            CU_eval_data = CU_X[test]
            eval_labels = Y[test]

            # tf-idf
            tfidf.fit(CU_train_data)
            CU_train_data = dense.transform(tfidf.transform(CU_train_data))
            CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))
            
            # standardization
            scaler.fit(CU_train_data)
            CU_train_data = scaler.transform(CU_train_data)
            CU_eval_data = scaler.transform(CU_eval_data)

            # normalization
            CU_train_data = normalize(CU_train_data)
            CU_eval_data = normalize(CU_eval_data)

            train_data =  CU_train_data
            eval_data = CU_eval_data

            # evaluation
            rbfsvm.fit(train_data, train_labels)
            lsvm.fit(train_data, train_labels)
            mlp.fit(train_data, train_labels)
            dTree.fit(train_data, train_labels)
            RF.fit(train_data, train_labels)
            KNN.fit(train_data, train_labels)

            rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
            lsvm_acc = lsvm.score(eval_data, eval_labels)
            mlp_acc = mlp.score(eval_data, eval_labels)
            dTree_acc = rbfsvm.score(eval_data, eval_labels)
            RF_acc = lsvm.score(eval_data, eval_labels)
            KNN_acc = mlp.score(eval_data, eval_labels)

            fold_accuracy.append((lsvm_acc,
                                  rbfsvm_acc,
                                  mlp_acc,
                                  dTree_acc,
                                  RF_acc,
                                  KNN_acc))
            print(lsvm_acc,"  ",
                  rbfsvm_acc,"  ",
                  mlp_acc,"  ",
                  dTree_acc,"  ",
                  RF_acc,"  ",
                  KNN_acc)
    print(('RBFSVM, LSVM,  MLP,  DTREE,  RF,  KNN'))
    print(np.mean(fold_accuracy, axis = 0))
    #---------------------------------------------------------------------------
    # Save the Trained Models Now
    path = "Trained_Models/"
    dump(lsvm, open(path + 'lsvm.pkl', 'wb'))
    dump(rbfsvm, open(path + 'rbfsvm.pkl', 'wb'))
    dump(mlp, open(path + 'mlp.pkl', 'wb'))
    dump(dTree, open(path + 'dTree.pkl', 'wb'))
    dump(RF, open(path + 'RF.pkl', 'wb'))
    dump(KNN, open(path + 'KNN.pkl', 'wb'))

    print('******************')
    print('Saved the Models')
    print('******************')


def preprocessVector(X):
    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = DenseTransformer()
    # tf-idf
    tfidf.fit(X)
    X = dense.transform(tfidf.transform(X))

    # standardization
    scaler.fit(X)
    X = scaler.transform(X)

    # normalization
    X = normalize(X)

    return X

def getPredictions(X):
    path   = "Trained_Models/"
    mlp    = load(open(path+'mlp.pkl',    'rb'))
    lsvm   = load(open(path+'lsvm.pkl',   'rb'))
    rbfsvm = load(open(path+'rbfsvm.pkl', 'rb'))
    dTree  = load(open(path+'dTree.pkl',  'rb'))
    RF     = load(open(path+'RF.pkl',     'rb'))
    KNN    = load(open(path+'KNN.pkl',    'rb'))

    p1 = mlp.predict(X[:,:])
    p2 = lsvm.predict(X[:,:])
    p3 = rbfsvm.predict(X[:,:])
    p4 = dTree.predict(X[:,:])
    p5 = RF.predict(X[:,:])
    p6 = KNN.predict(X[:,:])

    results = []
    for i in range(X.shape[0]):

        thisP = [p1[i], p2[i], p3[i], p4[i], p5[i], p6[i]]
        
      
        # intilize null lists
        unique_list = []
        counts = []

        for x in thisP: 
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x)
                counts.append(thisP.count(x))
        #print(unique_list, counts)
        r = thisP[np.argmax(np.array(counts))]
        results.append(r)

    return results