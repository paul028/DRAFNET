# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:21:37 2020

@author: Paul Vincent Nonat
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as p

file = p.read_csv('lorawan_antwerp_2019_dataset_withSF.csv') # doi 10.5281/zenodo.1212478
columns = file.columns
x = file[columns[0:72]] #RSSI DATA
SF=file[columns[73:74]]
SF=SF.values
x = -abs(x)
y = file[columns[75:]]
y_values = y.values

# clustering per closest gateway
m = p.DataFrame.idxmax(x,axis=1)
i = 1
indexing = dict()
for col in columns:
    indexing[col] = i
    i = i+1
colors = list()
for entry in m:
    colors.append(indexing[entry])  
    
plt.title('Antwerp Belgium LoRaWAN Device Ground Truth Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(y_values[:,1],y_values[:,0],c=colors,s=0.1, marker='.',cmap='hsv')
plt.savefig('figures/map.png', dpi=600, pad_inches=0)
plt.show()

a = x.values.flatten()
a = np.delete(a, np.where(a == [-200]), axis=0)
n, bins, patches = plt.hist(x=a, bins=25, normed=1, facecolor='b', alpha=0.8,rwidth=1,cumulative=0)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('RSSI Value')
plt.ylabel('Frequency')
plt.title('Histogram of RSSI values of all signal receptions')
plt.savefig('figures/histogram_rssi_values.png', dpi=600, bbox_inches='tight')
plt.show()
RSSI = x.values

counter = np.zeros(len(RSSI[0]))
for q in range(len(RSSI)):
    for w in range(len(RSSI[q])):
        if(RSSI[q][w]>-200):
            counter[w]= counter[w]+1

receiver_counter=np.zeros(len(RSSI))
for q in range(len(RSSI)):
    for w in range(len(RSSI[q])):
        if(RSSI[q][w]>-200):
            receiver_counter[q]=receiver_counter[q]+1

gateways=list()
for g in range(len(RSSI[0])):
    gateways.append(g+1)
    
gateways=np.array(gateways)

plt.bar(gateways,counter,0.35,color='blue')
plt.xlabel("Base Station")
plt.ylabel("Number of Message Received")
plt.title('Number of Message Received by LoRaWAN Base Station')
plt.savefig('figures/number_of_message_receved_basestation.png',dpi=600)
plt.show()
len(np.where(counter==0)[0]) ## 28 out of 72 gateways did not receive any message
np.where(counter==0)[0] #gateways without received message

receiver_sf=np.zeros((7,10))
for q in range(len(receiver_counter)):
    receiver_sf[SF[q][0]-7][int(receiver_counter[q]-1)]=receiver_sf[SF[q][0]-7][int(receiver_counter[q]-1)]+1

fig,ax = plt.subplots()
for z in range(len(receiver_sf)):
    if z==0:
        ax.bar([1,2,3,4,5,6,7,8,9,10],receiver_sf[z],label='SF'+str(z+7))
    else:
        ax.bar([1,2,3,4,5,6,7,8,9,10],receiver_sf[z],bottom=receiver_sf[z-1],label='SF'+str(z+7))
ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
ax.set_ylabel('No. of Message Received')
ax.set_xlabel('Number of Receiving Base Stations')
#ax.set_title('')
ax.legend()
plt.savefig('figures/dataset_dist.png',dpi=600)
plt.show()