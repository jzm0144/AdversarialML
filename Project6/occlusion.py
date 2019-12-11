import warnings
warnings.simplefilter('ignore')
import numpy as np
import ipdb as ipdb
from myAAS import *
import matplotlib.pyplot as plt
import pandas as pd

def getOcclusionMaps(X, model, c = 0):
    H = np.zeros((X.shape[0], X.shape[1], 25))
    mapNumber = H.shape[0]


    for i in range(mapNumber):
        print('outer loop ', i)
        thisX = X[i, :]
        H[i, :, :] = getHeatmap(thisX, model, c)
    print('---------------------')
    print('Occlusion Maps Ready ')
    print('---------------------')

    return H


def getHeatmap(thisX, model, c):
    W = np.zeros((95, 25))
    #W = np.zeros((np.shape(thisX)[1], 25))

    #probeY = ['1000','1001','1002','1003','1004','1005','1006','1007','1008','1009','1010',
    #          '1011','1012','1013','1014','1015','1016','1017','1018','1019','1020',
    #          '1021','1022','1023','1024']
    
    probeY = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    for i in range(len(probeY)):
        pY = probeY[i]
        W[:, i] = getH(thisX, model, pY, c)

    return W

def getH(thisX, model, pY, c):
    X = np.zeros((1, 95))
    #X = np.zeros(np.shape(thisX))

    #for i in range(thisX.shape[1]):
    for i in range(95):
        newX = np.array([thisX])
        newX[0, i] = c
        prob = model.predict_proba(newX[0:1, :])
        heat = prob[0,pY]
        X[0,i] = heat
    return X

def saveHeatmaps(H):
    for i in range(2500):

        df = pd.DataFrame(H[i, :, :])
        df.to_csv('allHeatmaps/'+str(i)+'.csv')


    HH = np.zeros((95, 25))

    for i in range(25):
        j = 100*(i+1)
        if i == 24:
            j = -1
        HH = sum(H[100*i:100*(i+1), :, :])
        HH  = HH/100

        df = pd.DataFrame(HH)
        name = '10'+str(i)
        if i <= 9:
            name = '100'+str(i)
        df.to_csv('avgHeatmaps/'+name+'.csv')
    print('--------------------------------------------------------------------')
    print("Single Heatmaps and Avg Heatmaps are Saved in your Current Directory")
    print('--------------------------------------------------------------------')


'''
X, Y = create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt')

X = preprocessVector(X)

#model = Train()
path   = "Trained_Models/"
model    = load(open(path+'mlp.pkl',    'rb'))


H = getOcclusionMaps(X[:, :], model, c = 0)

saveHeatmaps(H)

ipdb.set_trace()
'''