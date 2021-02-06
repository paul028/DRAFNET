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
import matplotlib.pyplot as plt

def get_exponential_distance(x,minimum,a=24):
	positive_x= x-minimum
	numerator = np.exp(positive_x.div(a))
	denominator = np.exp(-minimum/a)
	exponential_x = numerator/denominator
	exponential_x = exponential_x * 1000  #facilitating calculations
	final_x = exponential_x
	return final_x

def get_powed_distance(x,minimum,b=math.exp(1)):
	positive_x= x-minimum
	numerator = positive_x.pow(b)
	denominator = (-minimum)**(b)
	powed_x = numerator/denominator
	final_x = powed_x
	return final_x

a_list = list()
mean_val_error_list = list()
results = list()

for a in range(5,91,5):
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

	# temporary replacing the out-of-range value to find the true min
	x_train = x_train.replace(-200,0)
	minimum = int(x_train.min().min() - 1)
	print('minimum')
	print(minimum)
	x_train = x_train.replace(0,minimum)

	x_val = x_val.replace(-200,minimum)
	x_test = x_test.replace(-200,minimum)

	x_train = get_exponential_distance(x_train,minimum,a)
	x_val = get_exponential_distance(x_val,minimum,a)
	x_test = get_exponential_distance(x_test,minimum,a)	

	x_train = x_train.values
	x_val = x_val.values
	x_test = x_test.values
	y_train = y_train.values
	y_val = y_val.values
	y_test = y_test.values

	result = dict()
	reg_knn = KNeighborsRegressor(n_neighbors=5, metric='braycurtis', n_jobs=3)
	reg_knn.fit(x_train,y_train)
	y_predict_val_knn = reg_knn.predict(x_val)	
	y_predict_test_knn = reg_knn.predict(x_test)	

	val_mean_error = my_custom_haversine_error_stats(y_predict_val_knn,y_val)
	val_median_error = my_custom_haversine_error_stats(y_predict_val_knn,y_val,statistical_metric='median')	
	test_mean_error = my_custom_haversine_error_stats(y_predict_test_knn,y_test)
	test_median_error = my_custom_haversine_error_stats(y_predict_test_knn,y_test,statistical_metric='median')

	print(a)	
	print(val_mean_error)
	print(val_median_error)
	print(test_mean_error)
	print(test_median_error)
	print()
	result['a'] = a
	result['val_mean_error'] = val_mean_error
	result['val_median_error'] = val_median_error
	result['test_mean_error'] = test_mean_error
	result['test_median_error'] = test_median_error
	results.append(result)
	a_list.append(a)
	mean_val_error_list.append(val_mean_error)


plt.plot(a_list,mean_val_error_list)
plt.xlabel('Parameter value \u03B1 of exponential data representation')
plt.ylabel('Mean validation set error')
plt.savefig('figures/alpha.png', dpi=600, bbox_inches='tight')

