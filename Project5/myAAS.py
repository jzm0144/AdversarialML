from myShenanigans import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score

from ModelTraining import *

import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb as ipdb
from myShenanigans import *




X, Y = Get_Casis_CUDataset()

# Train the Mode, the models are already trained
# Train()

X = preprocessVector(X)

yHat = getPredictions(X)
print(yHat)

