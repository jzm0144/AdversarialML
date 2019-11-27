import numpy as np 
import pandas as pd
# Only for osX
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ipdb as ipdb
from math import *

dataset1 = pd.read_excel('Project3_Dataset_v1.xlsx')

#Dataset1
x1, y1, out1 = dataset1['x'], dataset1['y'], dataset1['out']


#Dataset2
dataset2 = dataset1.copy()
k = []
for out in dataset2['out'].values:
    if out > 0.5:
        k.append(1.0)
    else:
        k.append(-1.0)
dataset2['out'] = k
x2, y2, out2 = dataset2['x'], dataset2['y'], dataset2['out']

'''
fig1 = plt.figure()
ax_set1 = fig1.add_subplot(111, projection='3d')
ax_set1.scatter(x1, y1, out1, c='r', marker='^')
plt.show()

fig2 = plt.figure()
ax_set2 = fig2.add_subplot(111, projection='3d')
ax_set2.scatter(x2, y2, out2, c='r', marker='^')
plt.show()
'''
###############################################
#################### Part-4 ###################
###############################################

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

yTrainPred1_mse = 0
yTrainPred2_mse = 0
yTrainPred3_mse = 0
yTestPred1_mse  = 0
yTestPred2_mse  = 0
yTestPred3_mse  = 0

act_func = ['relu', 'tanh']

for _ in range(30):
    dataset1Vals = dataset1.values
    X = dataset1Vals[:,:2]
    Y  = dataset1Vals[:,2]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=15, shuffle=True)#, random_state=42

    # Make the Neural Network Regression Model
    model1 = MLPRegressor(hidden_layer_sizes=(4),          activation=act_func[0])
    model2 = MLPRegressor(hidden_layer_sizes=(4, 2),       activation=act_func[0])
    model3 = MLPRegressor(hidden_layer_sizes=(6, 4, 4, 2), activation=act_func[0])

    # Train the Neural Network
    model1.fit(xTrain, yTrain)
    model2.fit(xTrain, yTrain)
    model3.fit(xTrain, yTrain)

    yTrainPred1 = model1.predict(xTrain)
    yTestPred1  = model1.predict(xTest)
    yTrainPred1_mse += mean_squared_error(yTrain, yTrainPred1)
    yTestPred1_mse  += mean_squared_error(yTest , yTestPred1 )

    yTrainPred2 = model2.predict(xTrain)
    yTestPred2  = model2.predict(xTest)
    yTrainPred2_mse += mean_squared_error(yTrain, yTrainPred2)
    yTestPred2_mse  += mean_squared_error(yTest , yTestPred2 )


    yTrainPred3 = model3.predict(xTrain)
    yTestPred3  = model3.predict(xTest)
    yTrainPred3_mse += mean_squared_error(yTrain, yTrainPred3)
    yTestPred3_mse  += mean_squared_error(yTest , yTestPred3 )

print("H layers = 1    Training Error(mse): ", yTrainPred1_mse/30, "    Testing  Error(mse): ", yTestPred1_mse/30)
print("H layers = 2    Training Error(mse): ", yTrainPred2_mse/30, "    Testing  Error(mse): ", yTestPred2_mse/30)
print("H layers = 4    Training Error(mse): ", yTrainPred3_mse/30, "    Testing  Error(mse): ", yTestPred3_mse/30)


###############################################
#################### Part-3 ###################
###############################################
'''
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


dataset2Vals = dataset2.values
X2 = dataset2Vals[:,:2]
Y2 = dataset2Vals[:,2]
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(X2, Y2, test_size=15, shuffle=True)

svmLinear = SVC(kernel='linear')
svmRBF    = SVC(kernel='rbf')

svmLinear.fit(xTrain2, yTrain2)
svmRBF.fit(xTrain2, yTrain2)

yTrainPred_Linear = svmLinear.predict(xTrain2)
yTestPred_Linear  = svmLinear.predict(xTest2)

yTrainPred_RBF = svmRBF.predict(xTrain2)
yTestPred_RBF  = svmRBF.predict(xTest2)


print("SVM -- Linear Activation")
print("                                   Training Data")
print(confusion_matrix(     yTrain2, yTrainPred_Linear))
print(classification_report(yTrain2, yTrainPred_Linear))
print("                                   Testing Data")
print(confusion_matrix(     yTest2, yTestPred_Linear))
print(classification_report(yTest2, yTestPred_Linear))
print("\n\n\n\n\n---------------------------------")
print("SVM -- Radial Basis Activation")
print("                                   Training Data")
print(confusion_matrix(     yTrain2, yTrainPred_RBF))
print(classification_report(yTrain2, yTrainPred_RBF))
print("                                   Testing Data")
print(confusion_matrix(     yTest2, yTestPred_RBF))
print(classification_report(yTest2, yTestPred_RBF))
print("-------------------------------------------")


###############################################
#################### Part-1 ###################
###############################################

from sklearn.model_selection import train_test_split

class GRNN:
    def __init__(self, X, Y, sigma):
        self.X = X
        self.Y = Y
        self.sigma = sigma
        self.yHat = None

    def predict(self,x):
        dist = []
        w = []
        for i in range(self.X.shape[0]):
            distance = np.sum((x[:] - self.X[i,:])**2)
            dist.append(distance)

        twoSimgaSquare = 2*(self.sigma**2)
        for Di in dist:
            w.append(exp(-Di/twoSimgaSquare))

        denom = sum(w)
        numer = 0
        for i in range(len(w)):
            numer += w[i] * self.Y[i]

        self.yHat = numer/denom
        return self.yHat

    def MSE_TRAIN(self):
        preds = []
        for i in range(self.X.shape[0]):
            preds.append(self.predict(self.X[i,:]))

        preds = np.array(preds)
        error = sqrt(sum((preds - self.Y)**2))/preds.shape[0]
        return error

    def MSE_TEST(self, testX, testY):
        preds = []
        for i in range(testX.shape[0]):
            preds.append(self.predict(testX[i,:]))

        preds = np.array(preds)
        error = sqrt(sum((preds - testY)**2))/preds.shape[0]
        return error

mseTrain = 0
mseTest  = 0

for k in range(1):
    mySigma = k * 0.1+0.01
    mseSigma = 0
    for _ in range(30):
        dataset1Vals = dataset1.values
        X = dataset1Vals[:,:2]
        Y  = dataset1Vals[:,2]
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=15, shuffle=True)


        # For 85% train data we have xTrain, yTrain
        # for 15% test  data we have xTest , yTest
        myGRNN = GRNN(xTrain, yTrain, sigma=1.0)

        #for i in range(10):
        #    print("prediction: ", myGRNN.predict(xTest[i, :]), "expectation: ", yTest[i])

        mseTrain += myGRNN.MSE_TRAIN()
        mseTest  += myGRNN.MSE_TEST(xTest, yTest)
    mseSigma = (mseTrain/3 + mseTest/30)/2
    print("-------Sigma = ", mySigma,"----overall MSE = ",mseSigma)
    print("GRNN Train Data MSE = ", mseTrain/30)
    print("GRNN Test  Data MSE = ", mseTest/30)
    print("\n")
'''



