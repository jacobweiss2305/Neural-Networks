# -*- coding: utf-8 -*-
"""
Created on Sun May  5 08:36:51 2019

@author: jawei
"""

import numpy as np
import math



"""
Part 1.b.) Calculating the SVD and Psudeo Universe of A

Part 1.c.) Calculating least squares

"""
# Set up our Matrix problem Ax=b
A = np.array([[4, 2, 1],
              [3, 5, 2],
              [4, 1, 0],
              [2, 2, 4]])

b = np.array([14, 25, 9, 12])
 
#Compute the SVD of A
U, sigma, VT = np.linalg.svd(A)

#Compute psuedo-universe
np.linalg.pinv(A)

#Make a matrix sigma of the correct size
Sigma = np.zeros(A.shape)
Sigma[:3,:3] = np.diag(sigma)

#Check to make sure that we factorized A
(U.dot(Sigma).dot(VT) - A).round(4)

#Now define Sigma_pinv as the "pseudo-"inverse of Sigma
Sigma_pinv = np.zeros(A.shape).T
Sigma_pinv[:3,:3] = np.diag(1/sigma[:3])
Sigma_pinv.round(3)

#Now compute the SVD-based solution for the least-squares problem
x_svd = VT.T.dot(Sigma_pinv).dot(U.T).dot(b)
np.linalg.norm(A.dot(x_svd)-b, 2)
np.linalg.norm(x_svd)

"""
Part 2. D.) compute the vector [1 2 3 4] in terms of the new basis

"""

A = np.array([[0, 0, 0, 0],
              [0, (-5.0*math.sqrt(986))/493, 0, 0],
              [0, 0, math.sqrt(119.0)/493, 0],
              [0, 0, 0, (3.0*math.sqrt(14))/4]])

v = np.array([1,2,3,4])
A.dot(v)









