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

print(test.info())


print(train.shape)   # 10000 75
print(test.shape)    # 10000 71
print(y_predict.shape) # 10000 4
# train = train.interpolate() 
# test = test.interpolate()
# print(test.isnull().sum()[test.isnull().sum().values > 0])
x_train = train.iloc[:,:71]
print(x_train.head())
y_train = train.iloc[:,71:]
print(y_train.head())

plt.title('FUCKING')
plt.plot(x_train)
plt.xticks(x_train.ndray[:,])
plt.yticks(10000)
plt.show()
'''
# Nan 값 처리
train = train.fillna(method='bfill')
test = test.fillna(test.mean())
print(train.iloc[:,20:35])
# print(train.head(20))

# print(train[0].sum)
'''
'''
print('test_non : ',test.isnull().sum()[test.isnull().sum().values > 0])
print('train_non : ',train.isnull().sum()[train.isnull().sum().values > 0])
'''
'''
# train_zero = train[train != 0]
# print(train_zero)
# train = train.fillna(train.mean())
# print(train.head(10))


# train.filter(regex='_src$',axis=1).head(0).T.plot()
# plt.show()
train.filter(regex='_dst$',axis=1).head(6).T.plot()
plt.show()
test.filter(regex='_dst$',axis=1).head(6).T.plot()
plt.show()
'''