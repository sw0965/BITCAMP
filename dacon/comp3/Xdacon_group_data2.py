import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()

leaky = LeakyReLU(alpha = 0.2)
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10)

### 데이터 ###
x = pd.read_csv('./data/dacon/KAERI/train_features.csv',
                encoding = 'utf-8')
y = pd.read_csv('./data/dacon/KAERI/train_target.csv',
                index_col = 0, header = 0,
                encoding = 'utf-8')
x_pred = pd.read_csv('./data/dacon/KAERI/test_features.csv',
                     encoding = 'utf-8')
print(x.shape)                # (1050000, 6)
print(y.shape)                # (2800, 5)
print(x_pred.shape)           # (262500, 6)
'''
x_train = x
y_train = y
print(x_train.shape)        # (1050000, 6)
print(y_train.shape)        # (2800, 5)
print(x_train.head())


x_train = x_train.drop('Time', axis = 1)
print(x_train.head())


x_train = np.power(x_train.groupby(x_train['id']).mean(), 2)
print(x_train.shape)        # (2800, 4)

x_train.to_csv('./data/dacon/KAERI/x_train2.csv')


x_train = pd.read_csv('./data/dacon/KAERI/x_train2.csv',
                      index_col = 0, header = 0,
                      encoding = 'utf-8')
print(x_train.head())
print(x_train.shape)        # (2800, 4)

print(y_train.head())
print(y_train.shape)        # (2800, 4)

'''

x_pred = x_pred.drop('Time', axis = 1)
print(x_pred.head())
x_pred = np.power(x_pred.groupby(x_pred['id']).mean(), 2)
print(x_pred.shape)        # (700, 4)

x_pred = x_pred.values
x_pred = scaler.fit_transform(x_pred)
print(x_pred.shape)

np.save('./data/dacon/KAERI/x_pred_group.npy', arr=x_pred)


# np.save('./data/dacon/KAERI/x_train_group.npy', arr=x_train)
# np.save('./data/dacon/KAERI/x_test_pipe.npy', arr=y)
# np.save('./data/dacon/KAERI/x_pred_pipe.npy', arr=x_pred)
