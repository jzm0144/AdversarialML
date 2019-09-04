# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:25:41 2019

@author: Gerry Dozier
"""
import os
import random
import sys
import math
import matplotlib.pyplot as plt
# Only for osX
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
#import csv
#import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#from tqdm import *
#import ipdb as ipdb

class aGaussianKernel:
    def __init__(self, target, desired_output):
        self.target = target
        self.desired_output = desired_output     

    def fire_strength(self, q, sigma):
        sum_squared = 0.0
        for i in range(0,len(q)):
            sum_squared += math.pow((q[i] - self.target[i]),2.0)
        #print("FS Sum_Squared = " + str(sum_squared))
        return math.exp(-sum_squared/pow((2.0*sigma),2.0))
        
    def print_gaussian_kernel(self):
        print("Target = ", self.target, " Desired Output = " + str(self.desired_output))
    
        
        
class aSimpleNeuralNetwork:
    def __init__(self,dataset_file_name,num_of_inputs,num_of_outputs):
        self.dataset_file_name = dataset_file_name
        self.num_of_inputs = num_of_inputs
        self.num_of_outputs = num_of_outputs
        self.training_instance = []
        self.neuron = []
        self.sigma = 0
        self.tp = 0         # true positive
        self.tn = 0         # true negative
        self.fp = 0         # false positive
        self.fn = 0         # false negative
        
        with open(self.dataset_file_name, "r") as dataset_file:
            for line in dataset_file:
                line = line.strip().split(" ")
                self.training_instance.append([float(x) for x in line[0:]])
                
        for i in range(len(self.training_instance)):
            temp = aGaussianKernel(self.training_instance[i][:self.num_of_inputs],self.training_instance[i][self.num_of_inputs])
            self.neuron.append(temp)
    
    def set_sigma(self, sigma):
        self.sigma = sigma
        
    def distance_squared(self,x,y):
        dist_sqrd = 0
        for i in range(len(x)):
            dist_sqrd += (x[i] - y[i])**2
        return dist_sqrd    
    
    def train(self):
        dmax = 0
        dist_squared = 0
        for i in range(len(self.neuron)-1):
            for j in range((i+1),len(self.neuron)):
                dist_squared = self.distance_squared(self.neuron[i].target,self.neuron[j].target)
                if dmax < dist_squared:
                    dmax = dist_squared
        self.sigma = math.sqrt(dmax)
        #print("dmax =", self.sigma)
                
    def check(self,query):
        sum_fire_strength = 0
        sum_fire_strength_x_desired_output = 0
        for i in range(len(self.neuron)):
            the_fire_strength = self.neuron[i].fire_strength(query,self.sigma)
            sum_fire_strength_x_desired_output += the_fire_strength * self.neuron[i].desired_output
            sum_fire_strength += the_fire_strength
        if (sum_fire_strength == 0.0):
            sum_fire_strength = 0.000000001               # to prevent divide by zero
        return sum_fire_strength_x_desired_output/sum_fire_strength
 
    def test_model(self,number_of_test_cases):
        for i in range(number_of_test_cases):
            test_case = []
            sum_squared_error = 0
            x = random.uniform(-100.0,100.0)
            y = random.uniform(-100.0,100.0)
            test_case.append(x)
            test_case.append(y)
            
            x2y2 = x**2 + y**2
            SchafferF6 = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2
            
            test_instance_result = self.check(test_case)
            sum_squared_error += (test_instance_result - SchafferF6)**2
            self.calculate_statistics(test_instance_result,SchafferF6)
        return (sum_squared_error/number_of_test_cases)
    
    def calculate_statistics(self, test_instance_result, SchafferF6):
       # print("in calculate", test_instance_result, " ", SchafferF6)
        if ((test_instance_result > 0.5) and (SchafferF6 > 0.5)):
            self.tp += 1
        if ((test_instance_result <= 0.5) and (SchafferF6 <= 0.5)):
            self.tn += 1
        if ((test_instance_result > 0.5) and (SchafferF6 <= 0.5)):
            self.fp += 1
        if ((test_instance_result <= 0.5) and (SchafferF6 > 0.5)):
            self.fn += 1
           
    def plot_model(self, number_of_test_cases, lb, ub):
        test_case_x = []
        test_case_y = []
        test_case_z = []
        
        schafferF6_x = []
        schafferF6_y = []
        schafferF6_z = []
        
        for i in range(number_of_test_cases):
            x = random.uniform(lb,ub)
            y = random.uniform(lb,ub)
            
            test_case_x.append(x)
            test_case_y.append(y)
            schafferF6_x.append(x)
            schafferF6_y.append(y)
            
            test_case = []
            test_case.append(x)
            test_case.append(y)
            #print("test case: ",test_case)
            test_case_z.append(self.check(test_case))
            
            x2y2 = x**2 + y**2
            SchafferF6 = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2
            schafferF6_z.append(SchafferF6)
            
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(test_case_x,test_case_y,test_case_z)
        plt.title("Simple Neural Network")
        ax1.set_zlim3d(0.2,1.0)
        plt.show()
        
        fig = plt.figure()
        ax2 = fig.add_subplot(1,1,1,projection='3d')
        ax2.scatter(schafferF6_x,schafferF6_y,schafferF6_z)
        plt.title("SchafferF6")
        ax2.set_zlim3d(0.2,1.0)
        plt.show()
            
    def print_training_set(self):
        print(self.training_instance)
        print(len(self.training_instance))
    
    def print_neurons(self):
        for i in range(len(self.neuron)):
            self.neuron[i].print_gaussian_kernel()

    def print_statistics(self):
        accuracy  = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
        recall    = self.tp/(self.tp + self.fn + 0.00001)
        precision = self.tp/(self.tp + self.fp + 0.00001)
        f1        = 2*(precision * recall)/(precision + recall + 0.00001)
        print("Accuracy:  ", accuracy)
        print("Recall:    ", recall)
        print("Precision: ", precision)
        print("F1:        ", f1)

    def get_statistics(self, metric):
        print(self.tp + self.tn + self.fp + self.fn)
        accuracy  = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
        recall    = self.tp/(self.tp + self.fn + 0.00001)
        precision = self.tp/(self.tp + self.fp + 0.00001)
        f1        = 2*(precision * recall)/(precision + recall + 0.00001)
        stats = {"acc": accuracy,
                "rec": recall,
                "pre": precision,
                "f1" : f1}
        return stats[metric]


class dataset_generator:
    def __init__(self,filename,lb,ub):
        self.filename = filename
        self.lb = lb
        self.ub = ub
    
    def generate_dataset(self,number_of_training_instances):
        dataset_file = open(self.filename, 'w')
        for i in range (number_of_training_instances):
            x = random.uniform(self.lb,self.ub)
            y = random.uniform(self.lb,self.ub)
            x2y2 = x**2 + y**2
            dataset_file.write(str(x) + " " + str(y) + " " + str(0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2) + "\n")
        dataset_file.close()

def plot_results(sig_list, acc_list,rec_list,pre_list,f1_list):
    plt.figure(1)
    plt.subplot(121)
    plt.plot(sig_list, acc_list)
    plt.scatter(sig_list,acc_list) 
    plt.ylim(0,1.1)
    plt.xlabel("Sigma");
    plt.ylabel("Accuracy");

    plt.subplot(122)
    plt.plot(sig_list, f1_list) 
    plt.scatter(sig_list, f1_list) 
    plt.ylim(0,1.1)
    plt.xlabel("Sigma")
    plt.ylabel("F1 Score")

    plt.savefig('Result2.png')
    plt.show()

avg_Accuracy_list  = []
avg_Recall_list    = []
avg_Precision_list = []
avg_F1_list        = []
sigma_list = list(range(1,160,10)) #[0.2, 0.5, 0.9, 1.4, 1.8, 2.5, 3.0]+list(range(4,155,5))
for sig in sigma_list:
    avg_Accuracy = []
    avg_Recall   = []
    avg_Precision= []
    avg_F1       = []

    iters_for_Avg = 30
    for i in range(iters_for_Avg):
        #ipdb.set_trace()

        lower_bound = -100.0
        upper_bound =  100.0

        dg = dataset_generator('dataset.txt',lower_bound,upper_bound)
        dg.generate_dataset(1000)

        simple_neural_network = aSimpleNeuralNetwork("dataset.txt",2,1)

        simple_neural_network.train()
        simple_neural_network.set_sigma(sig/100.0)
        simple_neural_network.test_model(1000)
        #print("Model Test AMSE: ", simple_neural_network.test_model(1000))
        #simple_neural_network.print_statistics()
        #print(simple_neural_network.get_statistics())
        # normal check
        #lower_bound2 = lower_bound
        #upper_bound2 = upper_bound

        # adversarial check
        lower_bound2 = -100.0
        upper_bound2 =  100.0

        #simple_neural_network.plot_model(10000,lower_bound2,upper_bound2)
        avg_Accuracy.append(simple_neural_network.get_statistics("acc"))
        avg_Recall.append(simple_neural_network.get_statistics("rec"))
        avg_Precision.append(simple_neural_network.get_statistics("pre"))
        avg_F1.append(simple_neural_network.get_statistics("f1"))
        
        del simple_neural_network


    avg_Accuracy  = sum(avg_Accuracy)/len(avg_Accuracy)
    avg_Recall    = sum(avg_Recall)/len(avg_Recall)
    avg_Precision = sum(avg_Precision)/len(avg_Precision)
    avg_F1        = sum(avg_F1)/len(avg_F1)
    #print("Average Accuracy:  ", avg_Accuracy)
    #print("Average Recall:    ", avg_Recall)
    #print("Average Precision: ", avg_Precision)
    #print("Average F1:        ", avg_F1)
    avg_Accuracy_list.append(avg_Accuracy)
    avg_Recall_list.append(avg_Recall)
    avg_Precision_list.append(avg_Precision)
    avg_F1_list.append(avg_F1)

#ipdb.set_trace()
plot_results(sig_list=sigma_list,
             acc_list=avg_Accuracy_list,
             pre_list=avg_Precision_list,
             rec_list=avg_Recall_list,
             f1_list =avg_F1_list)
print("All Done!!")



