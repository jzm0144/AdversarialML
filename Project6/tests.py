import warnings
warnings.simplefilter('ignore')
import ipdb as ipdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from myAAS import *
import occlusion
import attack


#------ Part-1 Perturbation, Model Training/Testing and Reporting Normal Accuracy

# We will do the entire study using the larger, masked and perturbed dataset
X, Y = create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt')

# Train the Mode, the models are already trained
Train()

# Preprocess the entire dataset before doing anything else
X = preprocessVector(X)

# Get the Predictions
yHat_Normal = getPredictions(X)


#------- Part-2 Get the Occlusion Heatmaps for the Entire Perturbed Dataset
path   = "Trained_Models/"
model    = load(open(path+'mlp.pkl',    'rb'))

#hX = occlusion.getOcclusionMaps(X[:, :], model, c = 0)  #hX = (2500 x 95 x 25)
#occlusion.saveHeatmaps(hX)

topScores = 10;
R  = np.zeros((topScores,3))
for k in range(topScores):
    turns = 5
    mu_Normal = 0
    mu_Attack = 0
    for _ in range(turns):
        # topLocations = dictionary of top lcoations topLocations['class'][probeID]
        # class is the classification decision and the probeID (0,1,2,....24) is the
        # heatmap query
        topLocations = attack.getSalienceLocations(path = 'avgHeatmaps', topScores = k)

        #------ Part-3 Implement the Feature Contamination Attacks on all the examples
        newX = attack.contaminate(X, topLocations)

        #------ Part-4 Check Accuracy with the Contaminated Dataset
        yHat_Attacked = getPredictions(newX)

        normalAcc = calcAccuracy(Y, yHat_Normal)
        attackAcc = calcAccuracy(Y, yHat_Attacked)

        mu_Normal += normalAcc
        mu_Attack += attackAcc
    mu_Attack = mu_Attack/turns
    mu_Normal = mu_Normal/turns

    print('-----------------------------------------------------------')
    print('TopScore = ',k," --- Normal Acc = ",mu_Normal,'--- Attack Acc = ', mu_Attack)
    print('-----------------------------------------------------------')
    R[k,0] = k
    R[k,1] = mu_Normal
    R[k,2] = mu_Attack


plt.figure();plt.title('Performance Reduction');plt.plot(R[:,0], R[:,1:]);plt.xlabel('Num of Significant Features Replaced');plt.ylabel('Classification Accuracy');plt.show();
ipdb.set_trace()