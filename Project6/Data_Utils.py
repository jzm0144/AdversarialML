import os
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from Extractor.DatasetInfo import DatasetInfo


def get_dataset_name(filepath):
    fullname = os.path.basename(filepath)
    return os.path.splitext(fullname)[0]


def get_text_dataset(filename):
    X = []
    Y = []
    with open(filename, "r") as feature_file:
        for line in feature_file:
            line = line.strip().split(",")
            Y.append(line[0])

            X.append([float(x) for x in line[1:]])
    return np.array(X), np.array(Y)


def get_numpy_dataset(filename):
    with open(filename, "rb") as feature_file:
        content = np.load(feature_file)

    return np.array(content[1].tolist()), np.array(content[0].tolist())


dataset_reader = [get_text_dataset, get_numpy_dataset]


def get_fallback_dataset(filename, current):
    dataset_reader.pop(dataset_reader.index(current))
    dataset_reader.insert(0, current)

    result = None
    failed = False
    for reader in dataset_reader:
        try:
            result = reader(filename)
            if failed:
                print("Oops! reader " + reader.__name__ + " successfully read the data from " + filename)
            break
        except:
            failed = True
            print("Oops! reader " + reader.__name__ + " failed.")

    return result

def _get_dataset(filename, info=None):
    if info is None:
        dataset = get_dataset_name(filename)
        info = DatasetInfo(dataset, descriptor="auto").read()

    if isinstance(info, str):
        info = DatasetInfo(info, descriptor="auto").read()

    if info.get_feature_prop("is_numpy", False):
        return get_numpy_dataset  # get_numpy_dataset(filename)

    return get_text_dataset  # get_text_dataset(filename)


def get_dataset(filename, info=None):
    parser = _get_dataset(filename, info)
    return get_fallback_dataset(filename, parser)

class DenseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


