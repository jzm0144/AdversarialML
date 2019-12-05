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





class anIndividual:
    def __init__(self, specified_mask_length):
        self.mask = []
        self.fitness    = 0
        self.mask_length = specified_mask_length
        
    def randomly_generate(self):
        zero_count=random.randint(0,self.mask_length)
        one_count = self.mask_length - zero_count
        self.mask=[0]*zero_count + [1]*one_count
        random.shuffle(self.mask)
        self.fitness = 0
    
    def calculate_fitness(self,model):
        if self.mask.count(0)==self.mask_length:
            self.fitness =0
        else:
            CU_X, Y = Data_Utils.Get_Casis_CUDataset()
            X=self.applyMask(CU_X)
            self.fitness = self.myModels(X,Y,model)
        
    def applyMask(self, CU_X):
        X=[]
        for i in range(len(CU_X)):
            temp=[]
            for j in range(self.mask_length):
                if self.mask[j]==1:
                    temp.append(CU_X[i][j])
            #print(len(temp))
            X.append(temp)
        return(X)

    def print_individual(self, i):
        print("mask "+str(i) +": " + str(self.mask) + " Fitness: " + str(self.fitness))
       

    def myModels(self,X, y, model):

        CU_X=np.array(X)
        Y=np.array(y)


        # Creating the Classifier Instances
        rbfsvm = svm.SVC() # SVM with Radial Basis Activation
        lsvm = svm.LinearSVC() # SVM with Linear Activation
        mlp = MLPClassifier(max_iter=2000) # Multilayer Perceptron Activation
        

        # 4 Fold CrossValidation
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        fold_accuracy = []
        
        scaler = StandardScaler()
        tfidf = TfidfTransformer(norm=None)
        dense = Data_Utils.DenseTransformer()
        acc=0.0
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
        
            # Model Training and Testing
            if(model=="rbfsvm"):
                rbfsvm.fit(train_data, train_labels)
                acc += rbfsvm.score(eval_data, eval_labels)
            if(model=="lsvm"):
                lsvm.fit(train_data, train_labels)
                acc += lsvm.score(eval_data, eval_labels)
            if(model=="mlp"):
                mlp.fit(train_data, train_labels)
                acc += mlp.score(eval_data, eval_labels)
        return acc/4



class SSGA:
    def __init__(self, population_size, mask_length,model):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.mask_length = mask_length
        self.population = []
        self.model=model
        
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.mask_length)
            individual.randomly_generate()
            individual.calculate_fitness(self.model)
            self.population.append(individual)
    
    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness): 
                worst_fitness = self.population[i].fitness
                worst_individual = i
        return worst_individual
    
    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness
        
    def evolutionary_cycle(self):
        mom = random.randint(0,self.population_size-1)
        dad = random.randint(0,self.population_size-1)
        kid = self.get_worst_fit_individual()
        for j in range(self.mask_length):
            i=random.randint(0,1)
            if i==1:
                self.population[kid].mask[j] = self.population[mom].mask[j]
            else:
                self.population[kid].mask[j] = self.population[dad].mask[j]
        self.population[kid].calculate_fitness(self.model)
       
    def print_population(self):
        for i in range(self.population_size):
            self.population[i].print_individual(i)
    
    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return("Best Indvidual: "+str(best_individual)+" "+ ','.join(map(str,self.population[best_individual].mask))+ " Fitness: "+ str(best_fitness))
    


ChromLength = 95
MaxEvaluations = 350
PopSize = 30
model="lsvm"
#model="mlp"
#model="rbfsvm"
file=open(model+".txt","w")
for i in range(1):
    myGA = SSGA(PopSize,ChromLength,model)
    myGA.generate_initial_population()
    print(myGA.print_best_max_fitness())
    #myGA.print_population()
    for i in range(MaxEvaluations-PopSize+1):
        myGA.evolutionary_cycle()
        if (myGA.get_best_fitness() >= 0.95):
            break
    
    #print("\nFinal Population\n")
    #myGA.print_population()
    file.write(str(myGA.print_best_max_fitness()))
    myGA.print_best_max_fitness()
    print("Function Evaluations: " + str(PopSize+i))
