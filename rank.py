# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:28:50 2019

@author: jweiss
"""

import numpy as np
from matplotlib import pyplot as plt

rows, cols = (4,4)

randMatrix = [np.random.randint(11, size=(4, 4)) for i in range(10000)]
rank = [np.linalg.matrix_rank(randMatrix[i]) for i in range(len(randMatrix))]
plt.xlim([min(rank)-1, max(rank)+1])
plt.hist(rank, bins = 3, alpha=0.5)


