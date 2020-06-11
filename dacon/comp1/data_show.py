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
test  = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
y_predict  = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)

rho = train.iloc[:,0]
print(rho.shape) #(10000,)


x = train.iloc[:,:71]
print(x.head())
y = train.iloc[:,71:]
print(y.shape)    # hhb hbo2 ca na

#src dst 데이터 분리
x_src = x.iloc[:,1:36]
# print(x_src.info())
x_dst = x.iloc[:,36:71]
# print(x_dst.info())


plt.plot(x_src(10), rho)
plt.show()