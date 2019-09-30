"""
@version 09/20/2019
@author: Gerry Dozier
@Modified by: Janzaib Masood, Linyuan Zhang 

COMP6970_Group6
Project_1
Developing Genetic Algorithms (GA) for solving the Schaffer F6 function
(in fewer than 4000 function evaluations) on a total of 30 runs:
"""
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import ipdb as debugger

def Analyze_Results(thisCrossOverType="spx", dispFlag=False):
    print("START --------------  ",thisCrossOverType, "-------------------")
    dataPath = thisCrossOverType + "_Results_Log.csv"
    df = pd.read_csv(dataPath)

    Elitist_GA             = df['EGA'].values
    Steady_State_GA        = df['SSGA'].values
    Steady_Generational_GA = df['SGGA'].values
    mu_plus_mu_GA          = df['µ_plus_µ_GA'] .values

    print("Means  = ", np.mean(Elitist_GA), "   ", np.mean(Steady_State_GA), "   ", np.mean(Steady_Generational_GA), "   ", np.mean(mu_plus_mu_GA),"\n\n")



    (t, p) = stats.f_oneway(Elitist_GA, Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA)
    print("t value = ", t, '   and p value = ',p)



    (t, p) = stats.f_oneway(Elitist_GA, Steady_State_GA, Steady_Generational_GA)
    print("t value = ", t, '   and p value = ',p)



    (t, p) = stats.ttest_ind(Elitist_GA, Steady_State_GA)
    print("t value = ", t, '   and p value = ',p)
    

    print("END -------------- ------------------------------------------")
    # Generating the BoxPlot for Results
    data = [Elitist_GA, Steady_State_GA, Steady_Generational_GA, mu_plus_mu_GA]
    plt.boxplot(data, labels= ['Elitist','SSGA','SGGA','µ+µ'], showmeans=True)
    plt.title('Fitness values of the Genetic Algorithms')
    plt.ylabel("Best Fitnesses")
    plt.savefig(thisCrossOverType + "_Results_Log.png")
    if dispFlag == True:
        plt.show()



class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness = 0
        self.chromosome_length = specified_chromosome_length

    def randomly_generate(self,lb, ub):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.uniform(lb, ub))
        self.fitness = 0
    def add_chromosome(self, chromesome1, chromosome2):
        self.chromosome.append(chromesome1)
        self.chromosome.append(chromosome2)
        self.fitness = 0

    def calculate_fitness(self):
        x2y2 = self.chromosome[0]**2 + self.chromosome[1]**2
        self.fitness = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2

    def print_individual(self, i):
        print("Chromosome "+str(i) +": " + str(self.chromosome) + " Fitness: " + str(self.fitness))



class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub, crossover_type, algorithm):
        if (population_size < 2):
            print("Error: Population Size should be higher than 2")
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
        self.crossover_type = crossover_type
        self.methods = algorithm

    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate(self.lb,self.ub)
            individual.calculate_fitness()
            self.hacker_tracker_x.append(individual.chromosome[0])
            self.hacker_tracker_y.append(individual.chromosome[1])
            self.hacker_tracker_z.append(individual.fitness)
            self.population.append(individual)

    def add_new_population(self,chrome1, chrome2):
        individual = anIndividual(self.chromosome_length)
        individual.add_chromosome(chrome1, chrome2)
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

    def remove_worst_fit_individuals(self, count_of_individuals):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for _ in range(count_of_individuals):
            for i in range(self.population_size):
                if (self.population[i].fitness < worst_fitness):
                    worst_fitness = self.population[i].fitness
                    worst_individual = i

            del self.population[worst_individual]

    def tournament_selection_without_replacement(self, k): # Tournament Selection without replacement
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

    def Generation(self, k, LIMIT):
        count = 0
        temp_population = []


        if (self.methods == "µ_plus_µ_GA"):
            kids_list = []
            for index in range(self.population_size):
                mom = self.tournament_selection_without_replacement(k)
                dad = self.tournament_selection_without_replacement(k)
                kids = self.crossover(mom, dad)
                kids_list.append(kids)

            for thisKid in kids_list:
                self.add_new_population(thisKid[0], thisKid[1])

            self.remove_worst_fit_individuals(count_of_individuals = len(kids_list))

        if (self.methods == "EGA"):
            while (count < self.population_size):
                mom = self.tournament_selection_without_replacement(k)
                dad = self.tournament_selection_without_replacement(k)
                kids = self.crossover(mom, dad)
                kid_fit = self.offspring_fitness(kids[0], kids[1])
                if (kid_fit > LIMIT):
                    self.population[count].chromosome[0] = kids[0]
                    self.population[count].chromosome[1] = kids[1]
                    self.population[count].calculate_fitness()
                    count = count+1

        if (self.methods == "SSGA"):
            mom = self.tournament_selection_without_replacement(k)
            dad = self.tournament_selection_without_replacement(k)
            kids = self.crossover(mom, dad)
            insert = self.get_worst_fit_individual()
            self.population[insert].chromosome[0] = kids[0]
            self.population[insert].chromosome[1] = kids[1]
            self.population[insert].calculate_fitness()

        if (self.methods == "SGGA"):
            mom = self.tournament_selection_without_replacement(k)
            dad = self.tournament_selection_without_replacement(k)
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
            point = random.randint(0, self.chromosome_length - 1)
            for j in range(self.chromosome_length):
                if j <= point:
                    kid.chromosome[j] = self.population[mom].chromosome[j]
                else:
                    kid.chromosome[j] = self.population[dad].chromosome[j]

                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
                trash.append(kid.chromosome[j])

        if (self.crossover_type == "midx"):
            for j in range(self.chromosome_length):
                propm = random.randint(0 ,1)
                if propm == 1:
                    kid.chromosome[j] = (self.population[mom].chromosome[j] + self.population[dad].chromosome[j]) / 2
                    propm = 0
                else:
                    kid.chromosome[j] = self.population[mom].chromosome[j]
                    propm = 1

                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
                trash.append(kid.chromosome[j])

        if (self.crossover_type == "BLX_0.0"):
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
k = 2
PopSize = 10
mu_amt  = 0.01
runtimes = int((MaxEvaluations - PopSize) / PopSize)
LIMIT = 0.5
crossover_type = ["BLX_0.0", "midx", "spx"]
algorithm_list = ["EGA", "SSGA", "SGGA", "µ_plus_µ_GA"]

results = {"EGA":[],
           "SSGA": [],
           "SGGA": [],
           "µ_plus_µ_GA": []}

repeat = 30
for turn in range(repeat):
    for algo in algorithm_list:
        if algo == algorithm_list[0] or algo == algorithm_list[1] or algo == algorithm_list[3]:
            thisCrossOverType = crossover_type[1]
        else:
            thisCrossOverType = crossover_type[0]

        case1 = aSimpleExploratoryAttacker(PopSize,ChromLength,mu_amt,lb,ub, thisCrossOverType, algo)
        case1.generate_initial_population()
        #case1.print_population()

        run_count = 0
        for i in range(MaxEvaluations-PopSize+1):
            case1.Generation(k, LIMIT)
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


'''
#for thisCrossOverType in crossover_type:
    for turn in range(repeat):
        for algo in algorithm_list:
            case1 = aSimpleExploratoryAttacker(PopSize,ChromLength,mu_amt,lb,ub, thisCrossOverType, algo)
            case1.generate_initial_population()
            #case1.print_population()

            run_count = 0
            for i in range(MaxEvaluations-PopSize+1):
                case1.Generation(k, LIMIT)
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
    if os.path.exists(thisCrossOverType + "_Results_Log.csv"):
      os.remove(thisCrossOverType + "_Results_Log.csv")
    df.to_csv(thisCrossOverType + "_Results_Log.csv")

Analyze_Results(thisCrossOverType="spx", dispFlag=True)
Analyze_Results(thisCrossOverType="midx", dispFlag=True)
Analyze_Results(thisCrossOverType="BLX_0.0", dispFlag=True)

'''