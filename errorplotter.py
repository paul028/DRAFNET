# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:50:39 2020

@author: Paul Vincent Nonat
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as p


xval = np.arange(0.1, 4, 0.5) 
yval = np.exp(-xval) 
  
plt.errorbar(0.5, 0.45, xerr = 0.4, yerr = 0.5) 
plt.errorbar(1.5, 0.45, xerr = 0.4, yerr = 0.5)  
plt.title('matplotlib.pyplot.errorbar() function Example') 
plt.show() 