"""
@version 09/16/2019
@author: Gerry Dozier
@author: Linyuan Zhang, Janzaib Masood, 

COMP6970_Group6
project1
Develop the following Genetic Algorithms (GA) for solving the Schaffer F6 function
(in fewer than 4000 function evaluations) on a total of 30 runs:
"""

import numpy as geek
import random
import sys
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#import ipdb as debugger

class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness = 0
        self.chromosome_length = specified_chromosome_length
        #self.group = geek.zeros([3, population_size])

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
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub, cross_type, algorithm):
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
        self.crossover_type = cross_type
        self.algorithms = algorithm

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

    def get_best_fit_individual(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness > best_fitness):
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_individual

    def tournament_selection(self, k):
        count = 0
        bestParent = -99999999999.0
        best_index = 0
        #parent_index = []

        while (count < k):
            index = random.randint(0, self.population_size - 1)
            if self.population[index].fitness > bestParent:
                bestParent = self.population[index].fitness
                best_index = index
            count = count + 1
        return best_index

    def generation(self, k, threashold):
        #kid = self.get_worst_fit_individual()
        #self.temp = [0,0]
        count = 0
        temp_population = []
        #kids = []
        if (self.algorithms == "Elitist__Generational_GA"):
            while (count < self.population_size):
                mom = self.tournament_selection(k)
                dad = self.tournament_selection(k)
                kids = self.crossover(mom, dad)
                kid_fit = self.offspring_fitness(kids[0], kids[1])
                if (kid_fit > threashold):
                    self.population[count].chromosome[0] = kids[0]
                    self.population[count].chromosome[1] = kids[1]
                    self.population[count].calculate_fitness()
                    count = count+1

        elif (self.algorithms == "Steady_State_GA"):
            mom = self.tournament_selection(k)
            dad = self.tournament_selection(k)
            kids = self.crossover(mom, dad)
            insert = self.get_worst_fit_individual()
            self.population[insert].chromosome[0] = kids[0]
            self.population[insert].chromosome[1] = kids[1]
            self.population[insert].calculate_fitness()

        elif (self.algorithms == "Steady_Generational_GA"):
            mom = self.tournament_selection(k)
            dad = self.tournament_selection(k)
            kids = self.crossover(mom, dad)
            temp = []
            best_index = self.get_best_fit_individual()
            remove_index = random.randint(0, self.population_size-1)
            if (remove_index == best_index):
                kk = random.randint(0, self.population_size-1)
                self.population[kk].chromosome[0] = kids[0]
                self.population[kk].chromosome[1] = kids[1]
                self.population[kk].calculate_fitness()
            else:
                self.population[remove_index].chromosome[0] = kids[0]
                self.population[remove_index].chromosome[1] = kids[1]
                self.population[remove_index].calculate_fitness()


        elif (self.algorithms == "µ+µ_GA"):
            for index in range(self.population_size):
                mom = self.tournament_selection(k)
                dad = self.tournament_selection(k)
                kids = self.crossover(mom, dad)
                self.population[index].chromosome[0] = kids[0]
                self.population[index].chromosome[1] = kids[1]
                self.population[index].calculate_fitness()



    def offspring_fitness(self, x, y):
        x2y2 = x**2 + y**2
        return (0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2)

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
        print("Best Indvidual: ",str(i)," ", self.population[i].chromosome, " Fitness: ", str(best_fitness))
    
    def return_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        #print("Best Indvidual: ",str(i)," ", self.population[i].chromosome, " Fitness: ", str(best_fitness))
        return best_fitness



    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        plt.show()

    def crossover(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate(self.lb,self.ub)
        kid_fit = 0
        trash = []
        if (self.crossover_type == "spx"):
            s_point = random.randint(0, self.chromosome_length - 1)
            for j in range(self.chromosome_length):
                if j <= s_point:
                    kid.chromosome[j] = self.population[mom].chromosome[j]
                else:
                    kid.chromosome[j] = self.population[dad].chromosome[j]

                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
                trash.append(kid.chromosome[j])

        elif (self.crossover_type == "midx"):
            for j in range(self.chromosome_length):
                propm = random.uniform(0 ,1)
                if propm <= 1:
                    kid.chromosome[j] = (self.population[mom].chromosome[j] + self.population[dad].chromosome[j]) / 2
                else:
                    kid.chromosome[j] = self.population[mom].chromosome[j]

                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
                trash.append(kid.chromosome[j])

        elif (self.crossover_type == "BLX_0.0"):
            for j in range(self.chromosome_length):
                kid.chromosome[j] = random.uniform(self.population[mom].chromosome[j], self.population[dad].chromosome[j])
                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)

                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
                trash.append(kid.chromosome[j])
        return trash


ChromLength = 2
ub = 100.0
lb = -100.0
MaxEvaluations = 4000
plot = 0
k = 6
PopSize = 8
mu_amt  = 0.01
runtimes = int((MaxEvaluations - PopSize) / PopSize)
threashold = 0.5
cross_type = "BLX_0.0"
algorithm_list = ["Elitist__Generational_GA", "Steady_State_GA", "Steady_Generational_GA", "µ+µ_GA"]

results = {"Elitist__Generational_GA":[],
           "Steady_State_GA": [],
           "Steady_Generational_GA": [],
           "µ+µ_GA": []}

repeat = 30
for turn in range(repeat):
    for algo in algorithm_list:
        case1 = aSimpleExploratoryAttacker(PopSize,ChromLength,mu_amt,lb,ub, cross_type, algo)
        case1.generate_initial_population()
        #case1.print_population()

        run_count = 0
        for i in range(MaxEvaluations-PopSize+1):
            case1.generation(k, threashold)
            if (i % PopSize == 0):
                if (plot == 1):
                    case1.plot_evolved_candidate_solutions()
                #print("At Iteration: " + str(i))
                #case1.print_population()
            # terminator: best fitness = 0.99754
            if (run_count >= runtimes):
                break
            run_count = run_count+1

        #print("\nFinal Population\n")
        #case1.print_population()
        #case1.print_best_max_fitness()
        #print(case1.return_best_max_fitness()) # The fitness for each turn
        results[algo].append(case1.return_best_max_fitness())
        #print("Function Evaluations: " + str(PopSize+i))
        #case1.plot_evolved_candidate_solutions()

df = pd.DataFrame.from_dict(results)
if os.path.exists("Results_Log.csv"):
  os.remove("Results_Log.csv")
df.to_csv("Results_Log.csv")
