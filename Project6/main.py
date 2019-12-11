import warnings
warnings.simplefilter('ignore')
import ipdb as ipdb
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
print('-------------------------')
print("Noramal Predictions Ready");
print('-------------------------')


#------- Part-2 Get the Occlusion Heatmaps for the Entire Perturbed Dataset
path   = "Trained_Models/"
model    = load(open(path+'mlp.pkl',    'rb'))

hX = occlusion.getOcclusionMaps(X[:, :], model, c = 0)  #hX = (2500 x 95 x 25)
occlusion.saveHeatmaps(hX)

# topLocations = dictionary of top lcoations topLocations['class'][probeID]
# class is the classification decision and the probeID (0,1,2,....24) is the
# heatmap query
topLocations = attack.getSalienceLocations(path = 'avgHeatmaps', top = 10)

#------ Part-3 Implement the Feature Contamination Attacks on all the examples
newX = attack.contaminate(X, topLocations)

#------ Part-4 Check Accuracy with the Contaminated Dataset
yHat_Attacked = getPredictions(newX)
print('-------------------------')
print("Attacked Predictions Ready")
print('-------------------------')

ipdb.set_trace()

