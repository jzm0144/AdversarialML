import warnings
warnings.simplefilter('ignore')
import ipdb as ipdb

from myAAS import ask_the_ensemble, Train, getUnigramsFromTextFiles
ipdb.set_trace()



ipdb.set_trace()
getUnigramsFromTextFiles(data_dir = "./textfiles/", feature_set_dir = "./datasets/")
ipdb.set_trace()

# Create the result file and delete old data from it
file = open("AdversarialTestResults.txt","w"); file.close()

# Extract the Features from the AdversarialText.txt file

generateFeatures(inputFile = 'AdversarialTest.txt', outputFile = "AdvTestUnigram.txt")

# Please insert path to your text file containing the attacks
y = ask_the_ensemble(input_name = "AdvTestUnigram.txt")

print(y)

file = open("AdversarialTestResults.txt","a")
for item in y:
    # Write the result to the File
    file.write("\n"+item)
file.write("\n")
file.close()










