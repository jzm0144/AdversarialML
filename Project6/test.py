import numpy as np
from Extractor import Extractors

def readAdvText(filename = "AdversarialTest.txt"):
    filenames = []
    X = []
    Y = []
    with open(filename, "r") as feature_file:
        for line in feature_file:
            line = line.strip().split(",")
            filenames.append(line[0])

    return filenames


def generateFeatures(inputFile = 'AdversarialTest.txt', outputFile = "AdvTestUnigram.txt"):
    filenames = readAdvText(filename = inputFile)


filenames = readAdvText(filename = "AdversarialTest.txt")



print(file)