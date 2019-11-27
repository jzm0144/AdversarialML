# -*- coding: utf-8 -*-
"""
Created on Sunday Sept  29 19:31:41 2019

@author: Janzaib Masood
"""

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
        self.chromosome_length = specified_chromosome_length

    def random_fill_particle(self, lb, ub):
        for i in range(self.chromosome_length):
            ist_point = random.uniform(lb, ub)
            self.x.append(ist_point)
            self.p.append(ist_point)
            self.v.append(random.uniform(lb, ub))

    def calc_fitness(self):
        x2y2 = self.x[0]**2 + self.x[1]**2
        self.x_fitness = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2
        X2Y2 = self.p[0]**2 + self.p[1]**2
        self.p_fitness = 0.5 + (math.sin(math.sqrt(X2Y2))**2 - 0.5) / (1+0.001*X2Y2)**2

    def print_particle(self, i):
        print("Particle "+str(i+1) +": "+ str(self.p) + " Fitness: "+str(self.p_fitness))
        i = i

class myPSO: 
    def __init__(self, swarm_size, specified_chromosome_length, phi_1, phi_2, lb, ub, K):
        if (swarm_size < 2):
            print("Error: Population size must be greater than 2")
            sys.exit()
        self.swarm_size = swarm_size
        self.swarm = []
        self.chromosome_length = specified_chromosome_length
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
            particle = aParticle(self.chromosome_length)
            particle.random_fill_particle(self.lb,self.ub)
            # Calculate the fitnesses of Particle
            particle.calc_fitness()
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
            self.swarm[i].calc_fitness()

            if self.swarm[i].x_fitness > self.swarm[i].p_fitness:
                for gene in range(self.chromosome_length):
                    self.swarm[i].p[gene] = self.swarm[i].x[gene]
                self.swarm[i].p_fitness = self.swarm[i].x_fitness


            self.gBestP, self.gBestP_Fitness = self.get_best_p_fitness()
            if self.swarm[i].x_fitness > self.gBestP_Fitness:
                for gene in range(self.chromosome_length):
                    self.gBestP[gene] = self.swarm[i].x[gene]
                gBestP_Fitness = self.swarm[i].x_fitness

    def moveParticles(self, psoType):
        if psoType == "Sync":
            for i in range(self.swarm_size):
                for d in range(self.chromosome_length):
                    Pid = self.swarm[i].p[d]
                    Xid = self.swarm[i].x[d]
                    cognition = self.phi_1*random.random()*(Pid - Xid)
                    social    = self.phi_2*random.random()*(self.gBestP  - Xid)
                    self.swarm[i].v[d] += cognition + social
                    self.swarm[i].v[d] *= self.K
            for i in range(self.swarm_size):
                for d in range(self.chromosome_length):
                    self.swarm[i].x[d] += self.swarm[i].v[d]
        if psoType == "aSync":
            self.count += 1
            for i in range(self.swarm_size):
                self.swarm[i].calc_fitness()

                if self.swarm[i].x_fitness > self.swarm[i].p_fitness:
                    for gene in range(self.chromosome_length):
                        self.swarm[i].p[gene] = self.swarm[i].x[gene]
                    self.swarm[i].p_fitness = self.swarm[i].x_fitness


                self.gBestP, self.gBestP_Fitness = self.get_best_p_fitness()
                if self.swarm[i].x_fitness > self.gBestP_Fitness:
                    for gene in range(self.chromosome_length):
                        self.gBestP[gene] = self.swarm[i].x[gene]
                    gBestP_Fitness = self.swarm[i].x_fitness
                for d in range(self.chromosome_length):
                    Pid = self.swarm[i].p[d]
                    Xid = self.swarm[i].x[d]
                    cognition = self.phi_1*random.random()*(Pid - Xid)
                    social    = self.phi_2*random.random()*(self.gBestP  - Xid)
                    self.swarm[i].v[d] += cognition + social
                    self.swarm[i].v[d] *= self.K
                    self.swarm[i].x[d] += self.swarm[i].v[d]







ChromLength = 2
ub = 100.0
lb = -100.0
MaxEvaluations = 4000
plot = 0

phi_1 = 2.05
phi_2 = 2.05
SwarmSize = 30
phi = phi_1 + phi_2
K = 2/abs(2 - phi - math.sqrt(phi**2 - 4*phi))

PsoType = ["Sync", "aSync"]

results = {"Sync":[], "aSync":[]}


repeat = 30
for turn in range(repeat):
    for updateType in PsoType:
        # Generate the SWARM Instance
        mySwarm = myPSO(SwarmSize,ChromLength,phi_1, phi_2, lb,ub, K)

        # Generate all the Swarm Particless
        mySwarm.generate_particles()
        
        k = 1
        while True:
            if updateType == "Sync":
                # Calculate and Update the Fitnesses in X, P and Best_Particle_Fitness
                mySwarm.fitnessUpdateSync()

                # Update the positions in X, P, and V
                mySwarm.moveParticles(psoType = updateType)

                # Update the Tray with new Particles to display
                mySwarm.updateParticleTray()

                # Visualize Particles
                #mySwarm.print_summary()
                #mySwarm.plot_evolved_candidate_solutions()
            if updateType == "aSync":
                # Update the positions in X, P, and V
                mySwarm.moveParticles(psoType = updateType)

                # Update the Tray with new Particles to display
                mySwarm.updateParticleTray()

                # Visualize Particles
                #mySwarm.print_summary()
                #mySwarm.plot_evolved_candidate_solutions()


            k = k + 1
            if(k > MaxEvaluations-SwarmSize):
                break
        results[updateType].append(mySwarm.get_best_mean_fitness())


df = pd.DataFrame.from_dict(results)
filename = "Project2_Results_Log.csv"
if os.path.exists(filename):
  os.remove(filename)
df.to_csv(filename)