# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:31:41 2019

@author: Gerry Dozier
"""

import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb as debugger

#
#  A Simple Steady-State, Real-Coded Genetic Algorithm       
#
    
class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length
        
    def randomly_generate(self,lb, ub):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.uniform(lb, ub))
        self.fitness = 0
    
    def calculate_fitness(self):
        x2y2 = self.chromosome[0]**2 + self.chromosome[1]**2
        self.fitness = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2

    def print_individual(self, i):
        print("Chromosome "+str(i) +": " + str(self.chromosome) + " Fitness: " + str(self.fitness))
      
class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.lb = lb
        self.ub = ub
        self.mutation_amt = mutation_rate * (ub - lb)
        self.population = []
        self.hacker_tracker_x = []
        self.hacker_tracker_y = []
        self.hacker_tracker_z = []
        
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate(self.lb,self.ub)
            individual.calculate_fitness()
            self.hacker_tracker_x.append(individual.chromosome[0])
            self.hacker_tracker_y.append(individual.chromosome[1])
            self.hacker_tracker_z.append(individual.fitness)
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

    def tournament_selection(self, k):
        bestParentFitness = -9999999999999.0
        bestParent = self.population[0]

        selected = []
        best_index = 0
        
        t = self.population[:]
        population_size = len(t)
        for i in range(k):
            index = random.randint(0,population_size -1)
            thisParent = t[index]
            selected.append(thisParent)
            #debugger.set_trace()
            if thisParent.fitness > bestParentFitness:
                bestParent        = thisParent
                bestParentFitness = bestParent.fitness
            t.remove(t[index])
            population_size = len(t)

        #print("K selected  = ", selected)
        best_index = self.population.index(bestParent)
        #print("Best Index =  ", best_index)
        #print("Best Parent = ", bestParent)
        return best_index
        
    def evolutionary_cycle(self, cross_type, k): # add crossover arg and k for tournament
        #mom = random.randint(0,self.population_size-1) # modify to use tournament
        #dad = random.randint(0,self.population_size-1) # modify to use tournament
        mom = self.tournament_selection(k)
        dad = self.tournament_selection(k)

        kid = self.get_worst_fit_individual()
        for j in range(self.chromosome_length):
            self.population[kid].chromosome[j] = random.uniform(self.population[mom].chromosome[j],self.population[dad].chromosome[j])
            self.population[kid].chromosome[j] += self.mutation_amt * random.gauss(0,1.0)
            if self.population[kid].chromosome[j] > self.ub:
                self.population[kid].chromosome[j] = self.ub
            if self.population[kid].chromosome[j] < self.lb:
                self.population[kid].chromosome[j] = self.lb
        self.population[kid].calculate_fitness()
        self.hacker_tracker_x.append(self.population[kid].chromosome[0])
        self.hacker_tracker_y.append(self.population[kid].chromosome[1])
        self.hacker_tracker_z.append(self.population[kid].fitness)
       
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
        print("Best Indvidual: ",str(best_individual)," ", self.population[best_individual].chromosome, " Fitness: ", str(best_fitness))
    
    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        plt.show()


ChromLength = 2
ub = 100.0
lb = -100.0
MaxEvaluations = 4000
plot = 0
k = 5

PopSize = 10
mu_amt  = 0.0075

runTimes = int((MaxEvaluations - PopSize)/PopSize)

case2 = aSimpleExploratoryAttacker(PopSize,ChromLength,mu_amt,lb,ub)

case2.generate_initial_population()
case2.print_population()

cross_type = "Single Point Crossover"

for i in range(runTimes):
    case2.evolutionary_cycle(cross_type, k)
    case2.print_population()

print("\nFinal Population\n")
case2.print_population()
case2.print_best_max_fitness()
print("Function Evaluations: ", PopSize+1)
case2.plot_evolved_candidate_solutions()


'''
for i in range(MaxEvaluations-PopSize+1):
    simple_exploratory_attacker.evolutionary_cycle()
    if (i % PopSize == 0):
        if (plot == 1):
            simple_exploratory_attacker.plot_evolved_candidate_solutions()
        print("At Iteration: " + str(i))
        simple_exploratory_attacker.print_population()
    if (simple_exploratory_attacker.get_best_fitness() >= 0.99754):
        break

print("\nFinal Population\n")
simple_exploratory_attacker.print_population()
simple_exploratory_attacker.print_best_max_fitness()
print("Function Evaluations: " + str(PopSize+i))
simple_exploratory_attacker.plot_evolved_candidate_solutions()


'''


    