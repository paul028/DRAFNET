import pandas as p
import math
import numpy as np
from sklearn.model_selection import train_test_split 



file = p.read_csv('lorawan_antwerp_2019_dataset.csv') # doi 10.5281/zenodo.1212478
columns = file.columns
x = file[columns[0:72]]
x = -abs(x)
# x = x.join(file[columns[69]]) # LoRa spreading factor
y = file[columns[72:]]

random_state = 42
x_train, x_test_val, y_train, y_test_val = train_test_split(x.values, y.values, test_size=0.3, random_state=random_state)
x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)

x_train_df = p.DataFrame(x_train, columns=x.columns.values.tolist())
x_val_df = p.DataFrame(x_val, columns=x.columns.values.tolist())
x_test_df = p.DataFrame(x_test, columns=x.columns.values.tolist())
x_train_df.to_csv('files/x_train.csv')
x_val_df.to_csv('files/x_val.csv')
x_test_df.to_csv('files/x_test.csv')

y_train_df = p.DataFrame(y_train, columns=y.columns.values.tolist())
y_val_df = p.DataFrame(y_val, columns=y.columns.values.tolist())
y_test_df = p.DataFrame(y_test, columns=y.columns.values.tolist())
y_train_df.to_csv('files/y_train.csv')
y_val_df.to_csv('files/y_val.csv')
y_test_df.to_csv('files/y_test.csv')