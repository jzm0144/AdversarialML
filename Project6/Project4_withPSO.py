# -*- coding: utf-8 -*-
"""
Created on Sunday Sept  29 19:31:41 2019

@author: Janzaib Masood
"""



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
# Only for osX
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import ipdb as ipdb



#
#  Particle Swarm Optimization: Algorithm       
#
class aParticle:
    def __init__(self, specified_chromosome_length):
        self.x = []
        self.p = []
        self.v = []
        self.x_fitness = 0
        self.p_fitness = 0
        self.mask_length = specified_chromosome_length
        self.fitness = 0
        self.mask = []

    def random_fill_particle(self, lb, ub):
        zero_count=random.randint(0,self.mask_length)
        one_count = self.mask_length - zero_count
        self.mask=[0]*zero_count + [1]*one_count
        random.shuffle(self.mask)

        ist_point = self.mask
        for i in self.mask:
            self.x.append(ist_point)
            self.p.append(ist_point)
            self.v.append(random.uniform(lb, ub))


    def randomly_fill_particle(self):
        zero_count=random.randint(0,self.mask_length)
        one_count = self.mask_length - zero_count
        self.mask=[0]*zero_count + [1]*one_count
        random.shuffle(self.mask)
        self.fitness = 0


    def calc_fitness(self,model):
        CU_X, Y = Data_Utils.Get_Casis_CUDataset()
        X=self.applyMask(CU_X)
        self.x_fitness = self.myModels(X,Y,model)
        self.p_fitness = self.x_fitness
        
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





class myPSO: 
    def __init__(self, swarm_size, specified_mask_length, phi_1, phi_2, lb, ub, K):
        if (swarm_size < 2):
            print("Error: Population size must be greater than 2")
            sys.exit()
        self.swarm_size = swarm_size
        self.swarm = []
        self.mask_length = specified_mask_length
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.lb = lb
        self.ub = ub
        self.hacker_tracker_x = []
        self.hacker_tracker_y = []
        self.hacker_tracker_z = []
        self.gBestP = [-1, -1]
        self.gBestP_Fitness = -1
        self.K = K
        self.count = 0

    def generate_particles(self):
        for i in range(self.swarm_size):
            # Make a Particle
            particle = aParticle(self.mask_length)
            particle.random_fill_particle(self.lb,self.ub)
            # Calculate the fitnesses of Particle
            particle.calc_fitness(model)
            self.swarm.append(particle)
        self.updateParticleTray()

    def updateParticleTray(self):
        for i in range(self.swarm_size):
            # Fill the particle tray for the display
            self.hacker_tracker_x.append(self.swarm[i].p[0])
            self.hacker_tracker_y.append(self.swarm[i].p[1])
            self.hacker_tracker_z.append(self.swarm[i].p_fitness)

    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("The Particles in the End")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        plt.show()           

    def print_summary(self, particle_flag = False):
        if particle_flag == True:
            for i in range(self.swarm_size):
                self.swarm[i].print_particle(i)
        print("\n\n\nBest Fitness = ", mySwarm.gBestP_Fitness)
        print("Number of Fitness Evaulations = ", mySwarm.count)

    def get_best_p_fitness(self):
        best_p_fitness = -99999999999.0
        best_p_particle = -1
        for i in range(self.swarm_size):
            if self.swarm[i].p_fitness > best_p_fitness:
                best_p_fitness = self.swarm[i].p_fitness
                best_p_particle = i
        return [best_p_particle, best_p_fitness]

    def get_best_mean_fitness(self):
        best_p_fitness = -99999999999.0
        best_p_particle = -1
        fit = 0.0
        for i in range(self.swarm_size):
            fit += self.swarm[i].p_fitness
        return fit/self.swarm_size

    def fitnessUpdateSync(self):
        self.count += 1
        for i in range(self.swarm_size):
            self.swarm[i].calc_fitness(model)

            if self.swarm[i].x_fitness > self.swarm[i].p_fitness:
                for gene in range(self.mask_length):
                    self.swarm[i].p[gene] = self.swarm[i].x[gene]
                self.swarm[i].p_fitness = self.swarm[i].x_fitness


            self.gBestP, self.gBestP_Fitness = self.get_best_p_fitness()
            if self.swarm[i].x_fitness > self.gBestP_Fitness:
                for gene in range(self.mask_length):
                    self.gBestP[gene] = self.swarm[i].x[gene]
                gBestP_Fitness = self.swarm[i].x_fitness

    def moveParticles(self, psoType):
        if psoType == "Sync":
            for i in range(self.swarm_size):
                for d in range(self.mask_length):
                    Pid = self.swarm[i].p[d]
                    Xid = self.swarm[i].x[d]
                    cognition = self.phi_1*random.random()*(Pid - Xid)
                    social    = self.phi_2*random.random()*(self.gBestP  - Xid)
                    self.swarm[i].v[d] += cognition + social
                    self.swarm[i].v[d] *= self.K
            for i in range(self.swarm_size):
                for d in range(self.mask_length):
                    self.swarm[i].x[d] += self.swarm[i].v[d]
        if psoType == "aSync":
            self.count += 1
            for i in range(self.swarm_size):
                self.swarm[i].calc_fitness(model)

                if self.swarm[i].x_fitness > self.swarm[i].p_fitness:
                    for gene in range(self.mask_length):
                        self.swarm[i].p[gene] = self.swarm[i].x[gene]
                    self.swarm[i].p_fitness = self.swarm[i].x_fitness


                self.gBestP, self.gBestP_Fitness = self.get_best_p_fitness()
                if self.swarm[i].x_fitness > self.gBestP_Fitness:
                    for gene in range(self.mask_length):
                        self.gBestP[gene] = self.swarm[i].x[gene]
                    gBestP_Fitness = self.swarm[i].x_fitness
                for d in range(self.mask_length):
                    Pid = self.swarm[i].p[d]
                    Xid = self.swarm[i].x[d]
                    cognition = self.phi_1*random.random()*(Pid - Xid)
                    social    = self.phi_2*random.random()*(self.gBestP  - Xid)
                    self.swarm[i].v[d] += cognition + social
                    self.swarm[i].v[d] *= self.K
                    self.swarm[i].x[d] += self.swarm[i].v[d]







ChromLength = 95
ub = 10.0
lb = -10.0
MaxEvaluations = 500
plot = 0

phi_1 = 2.05
phi_2 = 2.05
SwarmSize = 30


phi = phi_1 + phi_2
K = 2/abs(2 - phi - math.sqrt(phi**2 - 4*phi))



model="lsvm"
#model="mlp"
#model="rbfsvm"
results = []


repeat = 1
for turn in range(repeat):
    # Generate the SWARM Instance
    mySwarm = myPSO(SwarmSize,ChromLength,phi_1, phi_2, lb,ub, K)

    # Generate all the Swarm Particless
    mySwarm.generate_particles()
    
    k = 1
    while True:
        # Calculate and Update the Fitnesses in X, P and Best_Particle_Fitness
        mySwarm.fitnessUpdateSync()

        # Update the positions in X, P, and V
        mySwarm.moveParticles(psoType = "Sync")

        # Update the Tray with new Particles to display
        mySwarm.updateParticleTray()

        # Visualize Particles
        #mySwarm.print_summary()
        #mySwarm.plot_evolved_candidate_solutions()

        k = k + 1
        if(k > MaxEvaluations-SwarmSize):
            break
    results.append(mySwarm.get_best_mean_fitness())





df = pd.DataFrame.from_dict(results)
filename = "Project4_Results_Log.csv"
if os.path.exists(filename):
  os.remove(filename)
df.to_csv(filename)