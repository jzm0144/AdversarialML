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
import ipdb as ipdb


CU_X, Y = Data_Utils.Get_Casis_CUDataset()



fold_accuracy = []

for repeat in range(10):
    #-----------------------------Classifiers----------------------------
    # SVM with Radial Basis Function
    rbfsvm = svm.SVC()
    # Linear SVM
    lsvm = svm.LinearSVC()
    # Multilayer Perceptron
    mlp = MLPClassifier(max_iter=2000)


    # Data Manipulation, Preprocessing, Training and Testing

    # 4-Fold CrossValidation with Shuffling
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

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

        rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        mlp_acc = mlp.score(eval_data, eval_labels)

        fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))


print(np.mean(fold_accuracy, axis = 0))
#---------------------------------------------------------------------------
