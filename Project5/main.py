import Data_Utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score

import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb as ipdb
from myAAS import ask_the_ensemble


# Please insert path to your text file containing the attacks
path = "/Users/jzm0144/Janzaib_Playground/AdversarialML/Project5/attacks.txt"

x = ask_the_ensemble(path, output_name = "result.txt")
print(x) # X has also been saved to your directory where the code is located











