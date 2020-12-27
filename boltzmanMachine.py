# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:43:39 2020

@author: jaweiss
"""
import numpy as np


def calcEnergy(initial_state, weights, random_node):
    return 1/(1 + (np.exp(np.dot(np.delete(initial_state.copy(), random_node), np.delete(weights.copy(), random_node)))/tempature))


tempature = 30
nodes = 3
weights = np.array([1, -2, 3])
initial_state = np.where(np.random.uniform(size = nodes) <= .5, 0, 1)


while tempature != 0:
    random_node = np.random.choice(np.arange(initial_state.size))
    if np.random.uniform() >= calcEnergy(initial_state, weights, random_node):
        initial_state[random_node] = 0
    else:
        initial_state[random_node] = 1

    tempature -= 1
    print(tempature)
    

#step 1

initial_state


weights = np.array([1, -2, 3])
states = np.array([1, 0, 1])

summ = []
for weight in weights:
    for i,j in enumerate(states):
        print(i)
        if i < (len(states)-1):
            print(states[i])
            print(states[i+1])
            
       summ.append(weight*states[i]*states[i+1])
    
np.dot(weights,states)




















calcEnergy(states, weights, 1)

truth_table = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])

