import subprocess
import time
import argparse

print("Running SF Normalize Experiment")
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+SF" --pca 0 --epoch 10000 --patience 300',shell=True)
time.sleep(2)
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+PCA=40+SF" --pca 40 --epoch 10000 --patience 300',shell=True)
time.sleep(2)
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+PCA=10+SF" --pca 10 --epoch 10000 --patience 300',shell=True)
time.sleep(2)
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+PCA=7+SF" --pca 7 --epoch 10000 --patience 300',shell=True)
time.sleep(2)
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+PCA=5+SF" --pca 5 --epoch 10000 --patience 300',shell=True)
time.sleep(2)
subprocess.call('start python MLP_withPCA+SF.py --trial-name "MLP+PCA=3+SF" --pca 3 --epoch 10000 --patience 300',shell=True)
time.sleep(2)

	#subprocess.call('start python webots_robotpos_randomizer.py',shell=True)
	#subprocess.call('start python agregator_nodepython_centerofmass.py',shell=True)