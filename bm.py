# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:07:38 2020

@author: jaweiss
"""

def distanceMatrix():
    return np.matrix([
    [0,  10, 20, 5,  18],
    [10, 0,  15, 32, 10],
    [20, 15, 0,  25, 16],
    [5,  32, 25, 0,  35],
    [18, 10, 16, 35, 0],
    ])

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
    if state[dupRowIdx,0] != 1 or state[dupRowIdx,-1] != 1:
        return False
    for colIdx in range(state.shape[1]):
        citiesVisitedAtOnce = np.sum(state[:, colIdx])
        if citiesVisitedAtOnce != 1:
            return False
    return True


class TSP:
    def __init__(self, distanceMatrix):   
        self.distanceMatrix = distanceMatrix #distance matrix as defined in the function above.
        self.cities = distanceMatrix.shape[0] #Counting the number of total cities
        self.steps = self.cities + 1 #Adding 1 so we can return home
        self.totalVisits = self.cities * self.steps   
        self.states = np.eye(self.steps) #intialize a tour matrix
        self.states = np.delete(self.states, self.cities, axis=0)   #we need to take out the tour home.
        bias = 3*np.max(self.distanceMatrix) #penalty > bias > 2*longest distance
        penalty = 3*bias
        self.weights = self.initializeWeights(penalty, bias)
        self.temperature = self.initializeTemp(penalty, bias)

    def initializeWeights(self, penalty, bias):
        weights = np.zeros((self.cities, self.steps, self.cities, self.steps))
        for city in range(self.cities):
            distances = self.distanceMatrix[city, :]
            for step in range(self.steps):
                currentWeightMatrix = weights[city, step]
                prevstep = step - 1 if step > 0 else self.steps - 1
                currentWeightMatrix[:, prevstep] = distances
                nextstep = step + 1 if step < self.steps - 1 else 0
                currentWeightMatrix[:, nextstep] = distances
                currentWeightMatrix[:, step] = penalty
                if step == 0:
                    currentWeightMatrix[city, :-1] = penalty
                elif step == self.cities:
                    currentWeightMatrix[city, 1:] = penalty
                else:
                    currentWeightMatrix[city, :] = penalty
                if step == 0 or step == self.cities:
                    currentWeightMatrix[city, 0] = -bias
                    currentWeightMatrix[city, self.cities] = -bias
                else:
                    currentWeightMatrix[city, step] = -bias
        return weights

    def initializeTemp(self, penalty, bias):
        return ((penalty * self.cities * self.steps) - bias) * 100

    def probState(self, city, tour, temperature):
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
            for _ in range(self.totalVisits**2):
                city = np.random.random_integers(0, self.cities-1, 1)[0]
                tour = np.random.random_integers(0, self.steps-1, 1)[0]
                stateProbability, deltaConsensus = self.probState(city, tour, self.temperature)
                if np.random.binomial(1, stateProbability) == 0:
                    changes += 1   # just used for printing status...
                    self.states[city, tour] = 1 - self.states[city, tour]
                    if tour == 0:
                        self.states[city, self.steps-1] = self.states[city, tour]
                    elif tour == self.steps-1:
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

