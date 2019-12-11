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

import Data_Utils
from Extractor.DatasetInfo import DatasetInfo
from Extractor.Extractors import BagOfWords, Stylomerty, Unigram, CharacterGram

def ask_the_ensemble(input_name = "AdversarialTest.txt"):

    X, Y = Get_Casis_CUDataset(input_name)

    # Train the Mode, the models are already trained
    # Train()

    X = preprocessVector(X)

    ipdb.set_trace()

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
    print('******************')    
    return xData, np.array(yData)



def Train():
    CU_X, Y = create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt')

    fold_accuracy = []

    for repeat in range(1): # it was 10 intitially
        #-----------------------------Classifiers------------------------
        
        # Multilayer Perceptron
        mlp = MLPClassifier(hidden_layer_sizes = (95,25),
                            activation = ('relu'),
                            max_iter=1000)

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

            # training
            mlp.fit(train_data, train_labels)

            # evaluation
            mlp_acc = mlp.score(eval_data, eval_labels)

            
    print('MLP Accuracy = ', mlp_acc)

    #---------------------------------------------------------------------------
    # Save the Trained Models Now
    path = "Trained_Models/"
    dump(mlp, open(path + 'mlp.pkl', 'wb'))
    print('***************************')
    print('Trained and Saved the Model')
    print('***************************')
    return mlp


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
    print('--------------------')
    print('Preprocessing Done!!')
    print('--------------------')
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
    return np.array(results)


def getUnigramsFromTextFiles(data_dir = "./textfiles/", feature_set_dir = "./datasets/"):
    extractor = Unigram(data_dir + "", "casis25")
    extractor.start()
    lookup_table = extractor.lookup_table
    print("Generated Lookup Table:")
    #print(lookup_table)
    if lookup_table is not False:
        print("'"+"', '".join([str("".join(x)).replace("\n", " ") for x in lookup_table]) + "'")

    # Get dataset information
    dataset_info = DatasetInfo("casis25_bow")
    dataset_info.read()
    authors = dataset_info.authors
    writing_samples = dataset_info.instances

    print("\n\nAuthors in the dataset:")
    print(authors)

    print("\n\nWriting samples of an author advText")
    print(authors["advText01"])

    print("\n\nAll writing samples in the dataset")
    print(writing_samples)

    print("\n\nThe author of the writing sample advText01")
    print(writing_samples["advText01"])

    generated_file = feature_set_dir + extractor.out_file + ".txt"
    data, labels = Data_Utils.get_dataset(generated_file)
    #print(labels[0], data[0])

def syncFeat_Attack(feature_set_dir = "./datasets/", attackFile = 'AdversarialTest.txt',out_file = 'ordered_feats.txt'):
    # Just to clear everything of out_file at first
    out = open(feature_set_dir + out_file, "w")
    out.close()


    with open(attackFile, "r") as attackFile:
        for aline in attackFile:
            saline = aline.strip().split(".")
            aname = saline[0]
            with open(feature_set_dir+"casis25_ncu.txt") as featFile:
                for bline in featFile:
                    sbline = bline.strip().split(",")
                    bname = sbline[0]
                    if bname == aname:
                        out = open(feature_set_dir + out_file, "a")
                        out.write(bline)
                        out.close()
                featFile.close()
        attackFile.close()

def calcAccuracy(y, yHat):
    same = 0
    for i in range(2500):
        if y[i] == yHat[i]:
            same +=1
    return same/2500
