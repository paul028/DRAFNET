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
import numpy


def residuals(p, data):  # Residuals function needed by kmpfit
   x, y = data           # Data arrays is a tuple given by programmer
   a, b = p              # Parameters which are adjusted by kmpfit
   return (y-(a+b*x))


#dataset 1 -128dB without SF, with SF
x = np.array([1,2,3,4,5,6,7])
#trial_name1=["MLP.csv","MLP+PCA=44.csv","MLP+PCA=40.csv","MLP+PCA=10.csv","MLP+PCA=7.csv","MLP+PCA=5.csv","MLP+PCA=3.csv"]
#trial_name1=["ext+PCA=3.csv","ext+PCA=5.csv","ext+PCA=7.csv","ext+PCA=10.csv","ext+PCA=40.csv","ext+PCA=44.csv","ext.csv"]
trial_name1=["KNN+PCA=3.csv","KNN+PCA=5.csv","KNN+PCA=7.csv","KNN+PCA=10.csv","KNN+PCA=40.csv","KNN+PCA=44.csv","KNN.csv"]

trial_name2=["KNN+PCA=3+SF.csv","KNN+PCA=5+SF.csv","KNN+PCA=7+SF.csv","KNN+PCA=10+SF.csv","KNN+PCA=40+SF.csv","KNN+PCA=44+SF.csv","KNN+SF.csv"]
#trial_name2=["MLP+SF.csv","MLP+PCA=44+SF.csv","MLP+PCA=40+SF.csv","MLP+PCA=10+SF.csv","MLP+PCA=7+SF.csv","MLP+PCA=5+SF.csv","MLP+PCA=3+SF.csv"]
#trial_name2=["ext+SF.csv","ext+PCA=44+SF.csv","ext+PCA=40+SF.csv","ext+PCA=10+SF.csv","ext+PCA=7+SF.csv","ext+PCA=5+SF.csv","ext+PCA=3+SF.csv"]

#dataset 2 Out of Range Dependent on SF
#trial_name3=["MLP_SFD.csv","MLP+PCA=44_SFD.csv","MLP+PCA=40_SFD.csv","MLP+PCA=10_SFD.csv","MLP+PCA=7_SFD.csv","MLP+PCA=5_SFD.csv","MLP+PCA=3_SFD.csv"]
#trial_name3=["ext_SFD.csv","ext+PCA=44_SFD.csv","ext+PCA=40_SFD.csv","ext+PCA=10_SFD.csv","ext+PCA=7_SFD.csv","ext+PCA=5_SFD.csv","ext+PCA=3_SFD.csv"]
trial_name3=["KNN+PCA=3_SFD.csv","KNN+PCA=5_SFD.csv","KNN+PCA=7_SFD.csv","KNN+PCA=10_SFD.csv","KNN+PCA=40_SFD.csv","KNN+PCA=44_SFD.csv","KNN_SFD.csv"]

#trial_name4=["MLP+SF_SFD.csv","MLP+PCA=44+SF_SFD.csv","MLP+PCA=40+SF_SFD.csv","MLP+PCA=10+SF_SFD.csv","MLP+PCA=7+SF_SFD.csv","MLP+PCA=5+SF_SFD.csv","MLP+PCA=3+SF_SFD.csv"]
#trial_name3=["ext_SFD.csv","ext+PCA=44_SFD.csv","ext+PCA=40_SFD.csv","ext+PCA=10_SFD.csv","ext+PCA=7_SFD.csv","ext+PCA=5_SFD.csv","ext+PCA=3_SFD.csv"]
trial_name4=["KNN+PCA=3+SF_SFD.csv","KNN+PCA=5+SF_SFD.csv","KNN+PCA=7+SF_SFD.csv","KNN+PCA=10+SF_SFD.csv","KNN+PCA=40+SF_SFD.csv","KNN+PCA=44+SF_SFD.csv","KNN+SF_SFD.csv"]
trial_name=[trial_name1,trial_name2,trial_name3,trial_name4]
df_list=list()
for w in range(len(trial_name)):
    df =list()
    for q in range(len(trial_name[w])):
        df.append(pd.read_csv("KNN_modified/"+trial_name[w][q],header=None).iloc[1:].values.astype('float64'))#.values.astype('float64')
    df_list.append(df) #[without,with][trial]



distance_error_list=list()
for w in range(len(trial_name)):
    distance_error=list()
    for z in range(len(df)):
        temp=list()
        for q in range(len(df_list[w][z])):
            temp.append(df_list[w][z][q][1])
            
        distance_error.append(temp)
    distance_error_list.append(distance_errerror)

#df=df.iloc[1:].values.astype('float64')
errors_mean_list=list()
errors_min_list=list()
errors_max_list=list()
errors_minmax_list=list()
for w in range(len(trial_name)):
    errors_mean=list()
    errors_min=list()
    errors_max=list()
    errors_minmax=list()
    for z in range(len(distance_error_list[w])):
        errors_mean.append(np.mean(distance_error_list[w][z]))
        errors_min.append(np.min(distance_error_list[w][z]))
        errors_max.append(np.max(distance_error_list[w][z]))    
        min_max=np.mean(distance_error_list[w][z])-np.min(distance_error_list[w][z])
        errors_minmax.append(min_max)
    errors_mean_list.append(errors_mean)
    errors_min_list.append(errors_min)
    errors_max_list.append(errors_max)
    errors_minmax_list.append(errors_minmax)
    

fig, ax = plt.subplots()
for w in range(len(errors_mean_list)):
    #ax.errorbar(x,errors_mean_list[w], yerr = errors_minmax_list[w], uplims=errors_max_list[w],lolims=errors_min_list[w],fmt = '.',marker='o')#,color='red')
    ax.scatter(x,errors_mean_list[w],marker=(w+5))
    ax.plot(x,errors_mean_list[w])
    #ax.set_xticks(x)


labels = [r'$3$',r'$5$',r'$7$',r'$10$',r'$40$',r'$44$',r'$72$']
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Positioning Error(m)")
ax.set_xlabel("Number of Receiving Base Station")
ax.set_title("Performance Comparison in Reduced Atwerp LoRaWAN Dataset")
ax.legend(['KNN+OOR1','KNN+OOR1+SF','KNN+OOR2','KNN+OOR2+SF'])
plt.savefig("KNN_model_performance_mean_modified.png",dpi=200,bbox_inches = 'tight')
plt.plot()

