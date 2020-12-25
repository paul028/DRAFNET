# LoRaWAN Localization using Data Rate Aware Fingerprinting Network  

Abstract: Localization is an essential in any Internet-of-Things (IoT) implementation to give meaning to its data. Low Power Wide Area Network (LPWAN) technology such as LoRaWAN enables GPS-free localization through its long-long range communication and low power consumption. Various ranging and fingerprinting techniques have been studied so far. However, they suffer high positioning errors due to the Received Signal Strength (RSSI) variation. In this study, a Data Rate Aware Fingerprinting Network for LoRaWAN localization is developed to mitigate RSSI variation using the Spreading Factor SF information from sensor nodes that preprocess the RSSI out of range data to improve the performance. A publicly available dataset that offers signal strength and SF of the sensor node is used and is split to train, validate, and test the neural network model. A reduced version of the dataset is also utilized wherein messages with less than three receiving base station are discarded. The fingerprinting model achieved a mean positioning error of 293.60 meters on the publicly available dataset and 225.57 meters in the reduced version.

# Dependencies
Ensure that the following libraries are installed before running the python.
Tensorflow
numpy
pandas
scikit-learn
haversine
# Training the DRAFNET  
To train the DRAFNET using the original antwerp lorawan dataset, run the following script:
python MLP_withPCA+SF --trial-name "trial name" --pca=[number of principal component 1-72] --epoch=[trainingepoch] --sf=[1-> sf input on , 2-> sf input off] --oor=[[0]-200dBm [1]-128dBm [2]SF dependent]

to train the DRAFNET using the reduced antwerp lorawan dataset, run
python MLP_withPCA+SF_dataset_modified --trial-name "trial name" --pca=[number of principal component 1-72] --epoch=[trainingepoch] --sf=[1-> sf input on , 2-> sf input off] --oor=[[0]-200dBm [1]-128dBm [2]SF dependent]

To run all the trials, run the script
python run_all_trials.py
For generating the evaluation graph run
# EVALUATION

Note, make sure that the .json and H5 file of the trial is available before running the evaluation

python model_evaluator_original.py --trial-name, -- model-name --pca, --sf,--oor for original dataset
or
python model_evaluator_modified.py --trial-name, -- model-name --pca, --sf,--oor for reduced dataset

to plot all the performance graph, run 
python graph_test.py

--trial-name ""MLP -- model-name "MLP"--pca, --sf,--oor
