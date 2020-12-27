# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 06:10:03 2019

@author: jweiss
"""
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    return x*(1-x)

#Feed Forard Neural Network
X = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

np.random.seed(1)
# initialize weights randomly with mean 0
w = 2*np.random.random((3,1)) - 1

for i in range(10):
    
    l0 = X
    l1 = sigmoid(np.dot(l0,w))
       
    #error rate
    error = y-l1
    delta = error * sigmoidDeriv(l1)
    
    #update weights
    w += np.dot(X.T,delta)
    

    print l1

l1
