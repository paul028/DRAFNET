# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:31:46 2020

@author: Paul Vincent Nonat
"""


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randrange, uniform
import random
trial_name=["baseline_paper.csv","Baseline-7layers.csv","MLP+PCA.csv","MLP+PCA=10.csv","MLP+PCA=7.csv","MLP+PCA=5.csv","MLP+PCA=3.csv"]


df =list()
for q in range(len(trial_name)):
    df.append(pd.read_csv(trial_name[q],header=None).iloc[1:].values.astype('float64'))#.values.astype('float64')


distance_error=list()
for z in range(len(df)):
    temp=list()
    for q in range(len(df[z])):
        temp.append(df[z][q][1])
        
    distance_error.append(temp)
#df=df.iloc[1:].values.astype('float64')

errors_mean=list()
errors_median=list()
errors_min=list()
errors_max=list()
x = np.array([1,2,3,4,5,6,7])
my_xticks = ['3L-MLP','7-MLP','7L-MLP+PCA=40','7L-MLP+PCA=10','7L-MLP+PCA=7','7L-MLP+PCA=5','7L-MLP+PCA=3']
for z in range(len(distance_error)):
    errors_mean.append(np.mean(distance_error[z]))
    errors_median.append(np.median(distance_error[z]))
    errors_min.append(np.min(distance_error[z]))
    errors_max.append(np.max(distance_error[z]))    



yerr = np.random.random(7)+20
z = np.linspace(0, 7, 7)


fig, ax = plt.subplots()
ax.errorbar(x,errors_mean, yerr = errors_mean, uplims=errors_max,lolims=errors_min,fmt = '.',marker='o',color='red')
ax.set_xticks(x)
ax.set_xticklabels(my_xticks,rotation=45)

ax.set_ylabel("Mean Positioning Error(m)")
ax.set_xlabel("MLP based Fingerprinting")
ax.set_title("Model Performance Comparison")
plt.savefig("model_performance.png",dpi=200,bbox_inches = 'tight')
plt.plot(),