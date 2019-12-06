import myShenanigans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
import ipdb as ipdb
import warnings
warnings.simplefilter('ignore')

#CU_X, Y = myShenanigans.Get_Casis_CUDataset()

CU_X, Y = myShenanigans.create_Preturbed_Dataset(inputFile = 'CASIS-25_CU.txt')

fold_accuracy = []

for repeat in range(4): # it was 10 intitially
    #-----------------------------Classifiers----------------------------
    # SVM with Radial Basis Function
    rbfsvm = svm.SVC()
    # Linear SVM
    lsvm = svm.LinearSVC()
    # Multilayer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes = (95,25),
                        activation = ('relu'),
                        max_iter=1000)
    # Decision Tree
    dTree = DecisionTreeClassifier(random_state=0)
    # Random Forests
    RF = RandomForestClassifier(random_state=0)
    # KNN, replaced my NB with KNN later. Naive Bayes
    KNN = KNeighborsClassifier(n_neighbors=3)

    # Data Manipulation, Preprocessing, Training and Testing

    # 4-Fold CrossValidation with Shuffling
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = myShenanigans.DenseTransformer()

    for train, test in skf.split(CU_X, Y):
        #train split
        CU_train_data = CU_X[train]
        train_labels = Y[train]
        
        #test split
        CU_eval_data = CU_X[test]
        eval_labels = Y[test]

        # tf-idf
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))
        
        # standardization
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)

        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)

        train_data =  CU_train_data
        eval_data = CU_eval_data

        # evaluation
        rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train_data, train_labels)
        mlp.fit(train_data, train_labels)
        dTree.fit(train_data, train_labels)
        RF.fit(train_data, train_labels)
        KNN.fit(train_data, train_labels)

        rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        mlp_acc = mlp.score(eval_data, eval_labels)
        dTree_acc = rbfsvm.score(eval_data, eval_labels)
        RF_acc = lsvm.score(eval_data, eval_labels)
        KNN_acc = mlp.score(eval_data, eval_labels)

        fold_accuracy.append((lsvm_acc,
                              rbfsvm_acc,
                              mlp_acc,
                              dTree_acc,
                              RF_acc,
                              KNN_acc))
        print(lsvm_acc,"  ",
              rbfsvm_acc,"  ",
              mlp_acc,"  ",
              dTree_acc,"  ",
              RF_acc,"  ",
              KNN_acc)
print(('RBFSVM, LSVM,  MLP,  DTREE,  RF,  KNN'))
print(np.mean(fold_accuracy, axis = 0))
#---------------------------------------------------------------------------
