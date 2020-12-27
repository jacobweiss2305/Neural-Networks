# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 06:25:10 2019

@author: jweiss
"""

import numpy as np


A = np.matrix('.6 .5; -.1 1.2')
U, S, V = np.linalg.svd(A)

b1 = np.array([100, 1000])
b11 = U.dot(b1)
y1 = np.linalg.inv(S)

b2 = V.dot(y1)
b22 = U.dot(b2)
y2 = np.linalg.inv(S).dot(b22)

b3 = V.dot(y2)
b33 = U.dot(b3)
y3 = np.linalg.inv(S).dot(b33)

b4 = V.dot(y3)
b44 = U.dot(b4)
y4 = np.linalg.inv(S).dot(b44)

b5 = V.dot(y3)
b55 = U.dot(b5)
y5 = np.linalg.inv(S).dot(b55)


print(
      str(b1) + '\n',
      str(b2) + '\n',
      str(b3) + '\n',
      str(b4) + '\n',
      str(b5) + '\n'      
      
      )