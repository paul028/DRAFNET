import subprocess
import time

print("Opening")
#extra trees original
subprocess.call('python extratress.py --trial-name "ext" --pca 0 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=44" --pca 44 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=40" --pca 40 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=5" --pca 5 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=3" --pca 3 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=10" --pca 10 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=7" --pca 7 --sf 0 --oor 1',shell=True)
time.sleep(0.2)


subprocess.call('python extratress.py --trial-name "ext+SF" --pca 0 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=44+SF" --pca 44 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=40+SF" --pca 40 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=10+SF" --pca 10 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=7+SF" --pca 7 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=5+SF" --pca 5 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=3+SF" --pca 3 --sf 1 --oor 1',shell=True)
time.sleep(0.2)

subprocess.call('python extratress.py --trial-name "ext_SFD" --pca 0 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=44_SFD" --pca 44 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=40_SFD" --pca 40 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=10_SFD" --pca 10 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=7_SFD" --pca 7 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=5_SFD" --pca 5 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=3_SFD" --pca 3 --sf 0 --oor 2',shell=True)
time.sleep(0.2)


subprocess.call('python extratress.py --trial-name "ext+SF_SFD" --pca 0 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=44+SF_SFD" --pca 44 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=40+SF_SFD" --pca 40 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=10+SF_SFD" --pca 10 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=7+SF_SFD" --pca 7 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=5+SF_SFD" --pca 5 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress.py --trial-name "ext+PCA=3+SF_SFD" --pca 3 --sf 1 --oor 2',shell=True)
time.sleep(0.2)

#extra trees modified
subprocess.call('python extratress_mod.py --trial-name "ext" --pca 0 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=44" --pca 44 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=40" --pca 40 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=5" --pca 5 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=3" --pca 3 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=10" --pca 10 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=7" --pca 7 --sf 0 --oor 1',shell=True)
time.sleep(0.2)


subprocess.call('python extratress_mod.py --trial-name "ext+SF" --pca 0 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=44+SF" --pca 44 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=40+SF" --pca 40 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=10+SF" --pca 10 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=7+SF" --pca 7 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=5+SF" --pca 5 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=3+SF" --pca 3 --sf 1 --oor 1',shell=True)
time.sleep(0.2)

subprocess.call('python extratress_mod.py --trial-name "ext_SFD" --pca 0 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=44_SFD" --pca 44 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=40_SFD" --pca 40 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=10_SFD" --pca 10 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=7_SFD" --pca 7 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=5_SFD" --pca 5 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=3_SFD" --pca 3 --sf 0 --oor 2',shell=True)
time.sleep(0.2)


subprocess.call('python extratress_mod.py --trial-name "ext+SF_SFD" --pca 0 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=44+SF_SFD" --pca 44 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=40+SF_SFD" --pca 40 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=10+SF_SFD" --pca 10 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=7+SF_SFD" --pca 7 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=5+SF_SFD" --pca 5 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python extratress_mod.py --trial-name "ext+PCA=3+SF_SFD" --pca 3 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
#original
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP" --pca 0 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=44" --pca 44 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=40" --pca 40 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=5" --pca 5 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=3" --pca 3 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=10" --pca 10 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=7" --pca 7 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)


subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+SF" --pca 0 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=44+SF" --pca 44 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=40+SF" --pca 40 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=10+SF" --pca 10 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=7+SF" --pca 7 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=5+SF" --pca 5 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=3+SF" --pca 3 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)

subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP_SFD" --pca 0 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=44_SFD" --pca 44 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=40_SFD" --pca 40 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=10_SFD" --pca 10 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=7_SFD" --pca 7 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=5_SFD" --pca 5 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=3_SFD" --pca 3 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)


subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+SF_SFD" --pca 0 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=44+SF_SFD" --pca 44 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=40+SF_SFD" --pca 40 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=10+SF_SFD" --pca 10 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=7+SF_SFD" --pca 7 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=5+SF_SFD" --pca 5 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF.py --trial-name "MLP+PCA=3+SF_SFD" --pca 3 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
#modified
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP" --pca 0 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=44" --pca 44 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=40" --pca 40 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=5" --pca 5 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=3" --pca 3 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=10" --pca 10 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=7" --pca 7 --epoch 10000 --patience 300 --sf 0 --oor 1',shell=True)
time.sleep(0.2)


subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+SF" --pca 0 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=44+SF" --pca 44 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=40+SF" --pca 40 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=10+SF" --pca 10 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=7+SF" --pca 7 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=5+SF" --pca 5 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=3+SF" --pca 3 --epoch 10000 --patience 300 --sf 1 --oor 1',shell=True)
time.sleep(0.2)

subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP_SFD" --pca 0 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=44_SFD" --pca 44 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=40_SFD" --pca 40 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=10_SFD" --pca 10 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=7_SFD" --pca 7 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=5_SFD" --pca 5 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=3_SFD" --pca 3 --epoch 10000 --patience 300 --sf 0 --oor 2',shell=True)
time.sleep(0.2)


subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+SF_SFD" --pca 0 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=44+SF_SFD" --pca 44 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=40+SF_SFD" --pca 40 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=10+SF_SFD" --pca 10 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=7+SF_SFD" --pca 7 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=5+SF_SFD" --pca 5 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)
subprocess.call('python MLP_withPCA+SF_dataset_modified.py --trial-name "MLP+PCA=3+SF_SFD" --pca 3 --epoch 10000 --patience 300 --sf 1 --oor 2',shell=True)
time.sleep(0.2)