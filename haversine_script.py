import pandas as pd
import numpy as np
import scipy as sp
import IPython
import sklearn
from haversine import haversine
from sklearn.model_selection import train_test_split
import statistics


def calculate_pairwise_error_list(ground_truth, predictions):
	distances = list()
	for i in range(0,len(ground_truth)):
		ground_truth_list = ground_truth[i].tolist()
		predict_list = predictions[i].tolist()
		h= haversine(tuple(ground_truth_list),tuple(predict_list))*1000  # multiplying by 1000 to transform from Km to m
		distances.append(h)
	return distances

def my_custom_haversine_mean_error(ground_truth, predictions):
	distances = calculate_pairwise_error_list(ground_truth, predictions)
	return statistics.mean(distances)


def my_custom_haversine_error_stats(ground_truth, predictions,statistical_metric='mean',percentile=50):
	distances = calculate_pairwise_error_list(ground_truth, predictions)
	if statistical_metric=="mean":
		return statistics.mean(distances)
	elif statistical_metric=="median":
		return statistics.median(distances)	
	elif statistical_metric=="percentile" and (percentile>=0 or percentile<=100):
		return np.percentile(distances,percentile)	
	else:
		return statistics.mean(distances)