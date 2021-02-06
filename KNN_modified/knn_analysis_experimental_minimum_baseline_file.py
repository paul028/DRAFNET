import pandas as p
import math
import numpy as np
import itertools as it
from haversine import *
from haversine_script import * 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time
import collections
import csv

def get_exponential_distance(x,minimum,a=60):
	positive_x= x-minimum
	numerator = np.exp(positive_x.div(a))
	denominator = np.exp(-minimum/a)
	exponential_x = numerator/denominator
	exponential_x = exponential_x * 1000  #facilitating calculations
	final_x = exponential_x
	return final_x

def get_powed_distance(x,minimum,b=1.1):
	positive_x= x-minimum	
	numerator = positive_x.pow(b)
	denominator = (-minimum)**(b)
	powed_x = numerator/denominator
	final_x = powed_x
	return final_x



# reading the data
x_train = p.read_csv('files/x_train.csv')
x_val = p.read_csv('files/x_val.csv')
x_test = p.read_csv('files/x_test.csv')
y_train = p.read_csv('files/y_train.csv')
y_val = p.read_csv('files/y_val.csv')
y_test = p.read_csv('files/y_test.csv')

# removing column 0 that is just an index
x_columns = x_train.columns
x_train = x_train[x_columns[1:]]
x_val = x_val[x_columns[1:]]
x_test = x_test[x_columns[1:]]

y_columns = y_train.columns
y_train = y_train[y_columns[1:]]
y_val = y_val[y_columns[1:]]
y_test = y_test[y_columns[1:]]

# Adjusting the minimum

# temporary replacing the out-of-range value to find the true min
x_train = x_train.replace(-200,0)
minimum = int(x_train.min().min() - 1)
print('minimum')
print(minimum)
x_train = x_train.replace(0,minimum)
x_val = x_val.replace(-200,minimum)
x_test = x_test.replace(-200,minimum)

# positive
x_train = x_train -minimum
x_val = x_val -minimum
x_test = x_test-minimum

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

my_metrics = ['euclidean',
  'manhattan' ,
  'chebyshev' ,
  'minkowski', 
  'hamming' ,
  'canberra',
  'braycurtis',
  'jaccard' ,
  'matching' ,
  'dice' ,
  'kulsinski',
  'rogerstanimoto' ,
  'russellrao',
  'sokalmichener',
  'sokalsneath'   ]

results = list()
result = collections.OrderedDict()

x_train = x_train.values
x_val = x_val.values
x_test = x_test.values
y_train = y_train.values
y_val = y_val.values
y_test = y_test.values

for my_metric in my_metrics:
    reg_knn = KNeighborsRegressor(n_neighbors=1, metric=my_metric, n_jobs=-1)
    reg_knn.fit(x_train,y_train)
    y_predict_val_knn = reg_knn.predict(x_val)
    y_predict_test_knn = reg_knn.predict(x_test)

    val_mean_error = my_custom_haversine_error_stats(y_predict_val_knn,y_val)
    val_median_error = my_custom_haversine_error_stats(y_predict_val_knn,y_val,statistical_metric='median')	
    test_mean_error = my_custom_haversine_error_stats(y_predict_test_knn,y_test)
    test_median_error = my_custom_haversine_error_stats(y_predict_test_knn,y_test,statistical_metric='median')

    result = collections.OrderedDict()
    result['metric'] = my_metric
    result['val_mean_error'] = val_mean_error
    result['val_median_error'] = val_median_error
    result['test_mean_error'] = test_mean_error
    result['test_median_error'] = test_median_error
    print(result)
    print()
    results.append(result)
    print(results)


    toCSV = results
    keys = toCSV[0].keys()
with open('results/lora_minimum_experimental_k_1.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)