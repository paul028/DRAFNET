# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:43:59 2020

@author: Paul Vincent Nonat
"""

#DRAF preprocessing experiment
#extratrees reimplementation using the extra trees model from
#[1] G. G. Anagnostopoulos and A. Kalousis,
# “A reproducible comparison of rssi fingerprinting 
#localization methods using LoRaWAN,” 
#2019 16th Work. Positioning, Navig. Commun. WPNC 2019, pp. 3–8, 2019.

import time
from haversine_script import *
import numpy as np
import random
import pandas as p
import math
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
#def get_exponential_distance(x,minimum,a=60):
#	positive_x= x-minimum
#	numerator = np.exp(positive_x.div(a))
#	denominator = np.exp(-minimum/a)
#	exponential_x = numerator/denominator
#	exponential_x = exponential_x * 1000  #facilitating calculations
#	final_x = exponential_x
#	return final_x
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import pickle# to save trained machine learning model

def get_powed_distance(x,minimum,b=1.1):
	positive_x= x-minimum
	numerator = positive_x.pow(b)
	denominator = (-minimum)**(b)
	powed_x = numerator/denominator
	final_x = powed_x
	return final_x

def get_powed_distance_np(x,minimum,b=1.1):
    positive_x= x-minimum
    numerator = pow(positive_x,b)
    denominator = (-minimum)**(b)
    powed_x = numerator/denominator
    final_x = powed_x
    return final_x


def generate_dataset(components,random_state,sf_n,oor_value):
    print("Creating Dataset")
    file = p.read_csv('lorawan_antwerp_2019_dataset_withSF.csv')
    columns = file.columns
    x = file[columns[0:72]]
    SF = file[columns[73:74]]
    y = file[columns[75:]]

    x= np.array(x)
    y=np.array(y)
    SF=np.array(SF)
    # delete rows with baase station less than 3
    delete_item=list()
    size = len(x)
    BS= len(x[0])
    for w in range(size):
        counter =0
        for q in range(BS):
            if x[w][q] >-200:
                counter = counter+1
        if counter <3:
          delete_item.append(w)
          print("Row",w,"Less than 3 Gateways")
          
    print(" Total Rows to delete: ",len(delete_item)," Remaining Rows: ",(size-len(delete_item)))
    x=np.delete(x,delete_item,axis=0)
    y=np.delete(y,delete_item,axis=0)
    SF=np.delete(SF,delete_item)
    
    if oor_value==0:
        final_x = get_powed_distance_np(x,-200)
        
    if oor_value==1: #set to -128dBm
        print("Set out of range value to -200dBm") #current experiment        
        for w in range(len(x)):
            for q in range(BS):
                if x[w][q]==-200:
                    x[w][q]=-128
                
        final_x = get_powed_distance_np(x,-128)
        
    if oor_value==2: #rescale according to SF
        print("Set out of range value according to SF")
        SF=np.array(SF)
        
        for q in range(len(SF)):
            print("Updating data",q+1)
            for w in range(len(x[q])):
                if x[q][w]==-200:
                    if SF[q]==7:
                        x[q][w]= -123
                    if SF[q]==8:
                        x[q][w]= -126
                    if SF[q]==9:
                        x[q][w]= -129
                    if SF[q]==10:
                        x[q][w]= -132                    
                    if SF[q]==11:
                        x[q][w]= -134.5                    
                    if SF[q]==12:
                        x[q][w]= -137

        final_x = get_powed_distance_np(x,-137)

    scaler_x = preprocessing.MinMaxScaler().fit(final_x)
    final_x = scaler_x.transform(final_x)
    
    scaler_y = preprocessing.MinMaxScaler().fit(y)
    y= scaler_y.transform(y)
    SF=SF.astype('float64')
    for q in range(len(SF)):
        SF[q]=float(SF[q]/12)
    
    if components >0:
        print("PCA enabled",components)
        pca = PCA(n_components =components) 
          
        final_x = pca.fit_transform(final_x) 
        explained_variance = pca.explained_variance_ratio_ 
    
        if sf_n>0:
            print("SF enabled")
            final_x =np.column_stack((final_x,SF))
            x_train, x_test_val, y_train, y_test_val = train_test_split(final_x, y, test_size=0.3, random_state=random_state)
            x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
            print(x_train.shape)
            print(x_val.shape)
            print(x_test.shape)
        else:
            print("SF disabled")
            x_train, x_test_val, y_train, y_test_val = train_test_split(final_x, y, test_size=0.3, random_state=random_state)
            x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
            print(x_train.shape)
            print(x_val.shape)
            print(x_test.shape)         
    
    else:
        final_x =np.column_stack((final_x,SF))
        x_train, x_test_val, y_train, y_test_val = train_test_split(final_x, y, test_size=0.3, random_state=random_state)
        x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
        print(x_train.shape)
        print(x_val.shape)
        print(x_test.shape)    

        if sf_n>0:
            print("SF enabled")
            final_x =np.column_stack((final_x,SF))
            x_train, x_test_val, y_train, y_test_val = train_test_split(final_x, y, test_size=0.3, random_state=random_state)
            x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
            print(x_train.shape)
            print(x_val.shape)
            print(x_test.shape)
        else:
            print("SF disabled")
            x_train, x_test_val, y_train, y_test_val = train_test_split(final_x, y, test_size=0.3, random_state=random_state)
            x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
            print(x_train.shape)
            print(x_val.shape)
            print(x_test.shape)         
            
    n_of_features = x_train.shape[1]
    print("Done Generating Dataset")
    return x_train,y_train,x_val,y_val,x_test,y_test,n_of_features,scaler_y


def train_extratrees(x_train ,y_train,x_val,y_val,x_test,y_test,scaler_y,trial_name,random_state):
    print("training extra trees")
    reg = ExtraTreesRegressor(n_estimators=100,max_depth=40, min_samples_split=14, min_samples_leaf=1,n_jobs=2, random_state=random_state)
    reg.fit(x_train, y_train)
    
    y_predict_in_train = reg.predict(x_train)
    y_predict_in_val = reg.predict(x_val)
    y_predict = reg.predict(x_test)
    

    y_predict = scaler_y.inverse_transform(y_predict)
    y_predict_in_train = scaler_y.inverse_transform(y_predict_in_train)
    y_predict_in_val = scaler_y.inverse_transform(y_predict_in_val)
    y_train = scaler_y.inverse_transform(y_train)
    y_val = scaler_y.inverse_transform(y_val)
    y_test = scaler_y.inverse_transform(y_test)

    print("Train set mean error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_train, y_train,'mean')))
    print("Train set median error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_train, y_train,'median')))
    print("Train set75th perc error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_train, y_train,'percentile',75)))
    print("Val set mean error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_val, y_val,'mean')))
    print("Val set median error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_val, y_val,'median')))
    print("Val set 75th perc.  error: {:.2f}".format(my_custom_haversine_error_stats(y_predict_in_val, y_val,'percentile',75)))
    print("Test set mean error: {:.2f}".format(my_custom_haversine_error_stats(y_predict, y_test,'mean')))
    print("Test set median error: {:.2f}".format(my_custom_haversine_error_stats(y_predict, y_test,'median')))
    print("Test set  75th perc. error: {:.2f}".format(my_custom_haversine_error_stats(y_predict, y_test,'percentile',75)))

    test_error_list = calculate_pairwise_error_list(y_predict,y_test)
    p.DataFrame(test_error_list).to_csv("extratress_modified/"+trial_name+".csv")
    print("Experiment completed!!!")
    y_predict_lat=list()
    y_predict_long=list()
    y_test_lat=list()
    y_test_long=list()
    for x in range(len(y_predict)):
       y_predict_lat.append(y_predict[x][0])
       y_predict_long.append(y_predict[x][1])
       y_test_lat.append(y_test[x][0])
       y_test_long.append(y_test[x][1])
       #plt.plot([y_predict[x][0],y_test[x][0]],[y_predict[x][1],y_test[x][1]],color='green')
    
    plt.scatter(y_predict_lat,y_predict_long,s=0.1, marker='.',color='red',label='Predicted Pos')
    plt.scatter(y_test_lat,y_test_long,s=0.1,marker='*',color='blue',label='Ground Truth Pos')
    plt.title(trial_name+' Predicted Postion Map in Reduced Antwerp LoraWan Dataset')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')   
    plt.legend()
    plt.savefig("extratress_modified/"+trial_name+'_predictedmap_original.png',bbox_inches='tight',dpi=600)


def save_model(trained_model,trial_name):
    filename = "extratress_modified/"+trial_name+'.sav'
    pickle.dump(trained_model, open(filename, 'wb'))


if __name__ == '__main__':

	#to train the model. python --trial-name "trial name" --pca=[number of principal component 1-72] --epoch=[trainingepoch] --sf=[1-> sf input on , 2-> sf input off] --oor=[[0]-200dBm [1]-128dBm [2]SF dependent]
    parser = argparse.ArgumentParser(description="--trial-name, --pca, --sf,--oor")
    parser.add_argument('--trial-name',type=str,required=True)
    parser.add_argument('--pca',type=int,default=0,help='Principal Component')
    parser.add_argument('--sf',type=int,default=0,help='Spreading Factor as input [0] off [1] on')
    parser.add_argument('--oor',type=int,default=0,help='RSSI Out of Range Values [0]-200dBm [1]-128dBm [2]SF dependent')
    args = parser.parse_args()
    components=args.pca
    trial_name=str(args.trial_name)
    sf_n=args.sf
    oor_value =args.oor


    random_state = 42
    os.environ['PYTHONHASHSEED'] = "42"
    np.random.seed(42)
    random.seed(42)
    
    x_train,y_train,x_val,y_val,x_test,y_test,n_of_features,scaler_y = generate_dataset(components,random_state,sf_n,oor_value)
    trained_model =train_extratrees(x_train ,y_train,x_val,y_val,x_test,y_test,scaler_y,trial_name,random_state)
    save_model(trained_model,trial_name)
