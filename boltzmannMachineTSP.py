# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:03:50 2020

@author: jaweiss
"""
import numpy as np
import math
import time
import collections
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

city_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}


def random_pattern(size):
    return np.random.randint(0,100,size*6)

def distanceMatrix(pattern):
    side_length = (int(math.sqrt(1 + 8 * len(pattern))) - 1) // 2 + 1
    assert (side_length * (side_length - 1)) // 2 == len(pattern), "Pattern length must be a triangular number."
    grid = [[0] * side_length for i in range(side_length)]
    position = 0
    for i in range(0, side_length - 1):
        for j in range(0, side_length - 1 - i):
            element = pattern[position]; position += 1
            grid[i][i + j + 1] = element
            grid[i + j + 1][i] = element 
    return np.array(grid)

def path(state):
    path = []
    for colIdx in range(state.shape[1]):
        path.append(np.argmax(state[:,colIdx]))
    return path

def pathDistance(distanceMatrix, path):
    distance = 0
    for index in range(len(path))[1:]:
        distance += distanceMatrix[path[index - 1], path[index]]
    return distance

def isPathValid(state):
    duplicateVisits = 0
    dupRowIdx = None
    for rowIdx in range(state.shape[0]):
        timesVisited = np.sum(state[rowIdx, :])
        if timesVisited != 1 and timesVisited != 2:
            return False
        if timesVisited == 2:
            duplicateVisits += 1
            dupRowIdx = rowIdx
    if duplicateVisits != 1:
        return False

##########____________________Boltzmann Machine____________________############
class TSP:
    def __init__(self, distanceMatrix):   
        self.distanceMatrix = distanceMatrix
        self.numCities = distanceMatrix.shape[0]
        self.tourSteps = self.numCities + 1
        self.numStates = self.numCities * self.tourSteps   
        self.states = np.eye(self.tourSteps)
        self.states = np.delete(self.states, self.numCities, axis=0)   
        bias = 2*np.max(self.distanceMatrix)
        penalty = 2*bias
        
        self.weights = self._initWeights(penalty, bias)
        self.temperature = self._initTemp(penalty, bias)

    def _initWeights(self, penalty, bias):
        weights = np.zeros((self.numCities, self.tourSteps, self.numCities, self.tourSteps))
        for city in range(self.numCities):
            distances = self.distanceMatrix[city, :]
            for tourStep in range(self.numCities+1):
                curWeights = weights[city, tourStep]
                prevTourStep = tourStep - 1 if tourStep > 0 else self.tourSteps - 1
                curWeights[:, prevTourStep] = distances
                nextTourStep = tourStep + 1 if tourStep < self.tourSteps - 1 else 0
                curWeights[:, nextTourStep] = distances
                curWeights[:, tourStep] = penalty
                if tourStep == 0:
                    curWeights[city, :-1] = penalty
                elif tourStep == self.numCities:
                    curWeights[city, 1:] = penalty
                else:
                    curWeights[city, :] = penalty
                if tourStep == 0 or tourStep == self.numCities:
                    curWeights[city, 0] = -bias
                    curWeights[city, self.numCities] = -bias
                else:
                    curWeights[city, tourStep] = -bias
        return weights

    def _initTemp(self, penalty, bias):
        return ((penalty * self.numCities * self.tourSteps) - bias) * 100

    def _stateProbability(self, city, tour, temperature):
        states = self.states.copy()
        state = self.states[city, tour]
        weights = self.weights[city, tour]
        states[city, tour] = (1 - state)
        weightEffect = np.sum(weights * states)
        biasEffect = weights[city, tour]
        activityValue = weightEffect + biasEffect
        deltaConsensus = ((1 - state) - state) * activityValue
        exponential = np.exp(-1 * deltaConsensus / temperature)
        probability = 1 / (1 + exponential)
        return probability, deltaConsensus

    def solve(self):
        lastValidState = self.states.copy()
        lowest_temp = 0.1
        highest_temp = self.temperature
        start = time.time()
        shortStart = None
        validHits = 0
        statesExplored = collections.defaultdict(int)
        changes = 0

        while self.temperature > lowest_temp:
            if shortStart == None:
                shortStart = time.time()
            for _ in range(self.numStates**2):
                city = np.random.random_integers(0, self.numCities-1, 1)[0]
                tour = np.random.random_integers(0, self.tourSteps-1, 1)[0]
                stateProbability, deltaConsensus = self._stateProbability(city, tour, self.temperature)
                if np.random.binomial(1, stateProbability) == 0:
                    changes += 1   # just used for printing status...
                    self.states[city, tour] = 1 - self.states[city, tour]
                    if tour == 0:
                        self.states[city, self.tourSteps-1] = self.states[city, tour]
                    elif tour == self.tourSteps-1:
                        self.states[city, 0] = self.states[city, tour]

                    if isPathValid(self.states):
                        lastValidState = self.states.copy()
                        statesExplored[str(lastValidState)] += 1
                        validHits += 1
            self.temperature *= 0.975
            if time.time()-shortStart > 1:
                shortStart = None
                elapsed = time.time() - start
                percentEst = (math.log(((highest_temp)/(self.temperature+1)))/math.log(highest_temp))*100
                secLeft = (100*(elapsed/percentEst)-elapsed)
                if secLeft > 0:
                    m, s = divmod(secLeft, 60)
                    h, m = divmod(m, 60)
                    eta = "%d:%02d:%02d" % (h, m, s)
                else:
                    eta = "0:00:00"
                    percentEst = 100
                dist = pathDistance(self.distanceMatrix, path(lastValidState))
                print("Temp:%-12s  PercentComplete:%-10s  ETA:%-s    Flips:%-10s  BestDist:%-7s DeltaConsensus:%-10s ValidStates:%d/%d " % ( "%.2f"%self.temperature, "%3.2f %%"%percentEst, eta, str(changes), str(dist),  str(deltaConsensus),  len(statesExplored.values()), sum(statesExplored.values())))
                changes = 0

        return path(lastValidState)





# def distanceMatrix():
#     return np.matrix([
#     [0,  10, 20, 5,  18],
#     [10, 0,  15, 32, 10],
#     [20, 15, 0,  25, 16],
#     [5,  32, 25, 0,  35],
#     [18, 10, 16, 35, 0],
#     ])