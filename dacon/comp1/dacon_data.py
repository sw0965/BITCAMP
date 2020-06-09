import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

############### 데이터 불러오기 ###############
train      = pd.read_csv('./data/dacon/bio/train.csv',index_col=0, header=0)
x_predict  = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
y_predict  = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)

train_data = train.interpolate() 
x_predict = x_predict.interpolate()
# train = train_data.interpolate() 


# print(train.info())
# print(x_predict.info())
# print(x_predict.isnull().sum())