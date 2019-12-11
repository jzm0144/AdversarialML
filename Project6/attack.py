import numpy as np
import pandas as pd
from scipy import stats
from occlusion import *
import matplotlib.pyplot as plt
from myAAS import *


def getSalienceLocations(path = 'avgHeatmaps', topScores = 0):
    dic = {'1000':{},'1001':{},'1002':{},'1003':{},'1004':{},
            '1005':{},'1006':{},'1007':{},'1008':{},'1009':{},
            '1010':{},'1011':{},'1012':{},'1013':{},'1014':{},
            '1015':{},'1016':{},'1017':{},'1018':{},'1019':{},
            '1020':{},'1021':{},'1022':{},'1023':{},'1024':{}}
    
    for i in range(25):
        name = '10'+str(i)
        if i <= 9:
            name = '100'+str(i)
        df = pd.read_csv(path+'/'+name+'.csv')
        df = df.values
        df = df[:,1:] # don't need the index column
        dic[name] = getLocs(df, topScores)
    return dic

def getLocs(hX, topScores):
    myList = []
    for i in range(25):
        q = hX[:,i]
        qLocs = getTop(q, topScores)
        myList.append(qLocs)
    return myList


def getTop(h, topScores): #H is 1D (,95), and return list of top indices
    locs = []
    for i in range(topScores):
        t = np.argmax(h)
        locs.append(t)
        h[t] = 0
    return locs

def contaminate(X, topLocations):
    names = ['1000','1001','1002','1003','1004',
             '1005','1006','1007','1008','1009',
             '1010','1011','1012','1013','1014',
             '1015','1016','1017','1018','1019',
             '1020','1021','1022','1023','1024']
    for ind in range(len(names)):

        name = names[ind]
        if ind <= 9:
            probe = int(name[-1])
        else:
            probe = int(name[2:])
        # We want the top heat locations of only correct decisions
        cLocs = topLocations[name][probe]

        means, modes = getStats(X, cLocs, probe)

        # Upon each iteration it contaminates the instance of a particular class
        # Update X, slice by slice
        X = doChanges(X, means, modes, cLocs, classID=probe)
    print('-----------------------')
    print('Data Contamination Done')
    print('-----------------------')
    return X

def getStats(X, cLocs, probe):
    i = probe*100
    j = (probe+1)*100
    if j == 2500:
        j = -1

    thisX = X[i:j, :]
    means = []
    modes = []
    for thisLoc in cLocs:
        mean = np.mean(thisX[:,thisLoc])
        means.append(mean)

        m = stats.mode(thisX[:,thisLoc])
        modes.append(m[0][0])
    return means, modes

def doChanges(X, means, modes, cLocs,classID):
    i = classID*100
    j = (classID+1)*100
    if j == 2500:
        j = -1

    thisX = X[i:j, :]

    for ind in range(len(cLocs)):
        thisLoc = cLocs[ind]
        mean = means[ind]
        mode = modes[ind]
        Q = round(0.75*mean + 0.25*mode)
        R = round(Q - 1)
        thisX[:,thisLoc] = R
    X[i:j, :] = thisX
    return X