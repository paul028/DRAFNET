import time
from haversine_script import *
import numpy as np
import tensorflow as tf
import random
import pandas as p
import math
import matplotlib.pyplot as plt
import os
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 } ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
tf.debugging.set_log_device_placement(True)

#def get_exponential_distance(x,minimum,a=60):
#	positive_x= x-minimum
#	numerator = np.exp(positive_x.div(a))
#	denominator = np.exp(-minimum/a)
#	exponential_x = numerator/denominator
#	exponential_x = exponential_x * 1000  #facilitating calculations
#	final_x = exponential_x
#	return final_x

def get_powed_distance(x,minimum,b=1.1):
	positive_x= x-minimum
	numerator = positive_x.pow(b)
	denominator = (-minimum)**(b)
	powed_x = numerator/denominator
	final_x = powed_x
	return final_x


def generate_dataset(components,random_state):
    file = p.read_csv('lorawan_antwerp_2019_dataset_withSF.csv')
    columns = file.columns
    x = file[columns[0:72]]
    SF = file[columns[73:74]]
    y = file[columns[75:]]

    x = x.replace(-200,200) 
    minimum = x.min().min() - 1
    x = x.replace(200,minimum) #set dataset -200 to next posible minimum
    print('minimum')
    print(minimum)
    
    final_x = get_powed_distance(x,minimum)
    scaler_x = preprocessing.MinMaxScaler().fit(final_x)
    final_x = scaler_x.transform(final_x)
    
    scaler_y = preprocessing.MinMaxScaler().fit(y)
    y= scaler_y.transform(y)
    
    scaler_sf= preprocessing.MinMaxScaler().fit(SF)
    SF=scaler_sf.transform(SF)
    
    if components >0:
        pca = PCA(n_components =components) 
          
        x_train = pca.fit_transform(final_x) 
        explained_variance = pca.explained_variance_ratio_ 
    
        final_x =np.column_stack((final_x,SF))
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
    
    n_of_features = x_train.shape[1]
    return x_train,y_train,x_val,y_val,x_test,y_test,n_of_features,scaler_y


def create_model(n_of_features,dropout,l2,lr,random_state):
    
    model = Sequential()
    model.add(Dense(units=1024, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=1024, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=1024, input_dim=n_of_features, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=256, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(units=128, kernel_regularizer=regularizers.l2(l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(dropout))
    model.add(Dense(units=2))
    model.compile(loss='mean_absolute_error',optimizer=Adam(lr=lr))
    return model;

def train(x_train, y_train,x_val,y_val,epoch,batch_size,patience):
    cb =[EarlyStopping(monitor='val_loss', patience=patience, verbose =1, restore_best_weights=True)]
    history = model.fit(x_train, y_train,validation_data=(x_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1, callbacks= cb)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('training_curves.png')
    plt.show()
    trained_model= model
    return trained_model

def validate_model(trained_model, x_train ,y_train,x_val,y_val,x_test,y_test,scaler_y,trial_name):
    model=trained_model
    y_predict = model.predict(x_test, batch_size=batch_size) 
    y_predict_in_val = model.predict(x_val, batch_size=batch_size)
    y_predict_in_train = model.predict(x_train, batch_size=batch_size)
    
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
    p.DataFrame(test_error_list).to_csv(trial_name+".csv")
    print("Experiment completed!!!")


def save_model(trained_model,trial_name):
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.models import load_model
    
    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    model_json = model.to_json()
    
    
    with open(trial_name+".json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights(trial_name+".h5")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="--trial-name, --pca, --epoch,--patience")
    parser.add_argument('--trial-name',type=str,required=True)
    parser.add_argument('--pca',type=int,default=0,help='Principal Component')
    parser.add_argument('--epoch',type=int,default=100,help='Number of training epoch')
    parser.add_argument('--patience',type=int,default=300,help='Training patience')
    
    args = parser.parse_args()
    components=args.pca
    trial_name=str(args.trial_name)
    epoch=args.epoch
    patience=args.patience
   
    dropout = 0.15
    l2 = 0.00
    lr = 0.0005
    batch_size= 512

    random_state = 42
    os.environ['PYTHONHASHSEED'] = "42"
    np.random.seed(42)
    tf.random.set_random_seed(42)
    random.seed(42)
    
    x_train,y_train,x_val,y_val,x_test,y_test,n_of_features,scaler_y = generate_dataset(components,random_state)
    model=create_model(n_of_features,dropout,l2,lr,random_state)
    trained_model = train(x_train, y_train,x_val,y_val,epoch,batch_size,patience)
    validate_model(trained_model, x_train ,y_train,x_val,y_val,x_test,y_test,scaler_y,trial_name)
    save_model(trained_model,trial_name)