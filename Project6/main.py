import warnings
warnings.simplefilter('ignore')
import ipdb as ipdb

from myAAS import *


getUnigramsFromTextFiles(data_dir = "./textfiles/", feature_set_dir = "./datasets/")

syncFeat_Attack(feature_set_dir = "./datasets/", attackFile = 'AdversarialTest.txt',out_file = 'ordered_feats.txt')


# Please insert path to your text file containing the attacks
y = ask_the_ensemble(input_name = "./datasets/ordered_feats.txt")

print("The predictions List") ; print(y)


# Delete old data from the Results File
file = open("AdversarialTestResults.txt","w"); file.close()



attackFile = open('AdversarialTest.txt','r')
result = open("AdversarialTestResults.txt","a");
t = 0;
for line in attackFile:
    result.write(line[:13] + ',' +y[t]+'\n')
    t += 1

attackFile.close()
result.close()