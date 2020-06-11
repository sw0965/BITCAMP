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
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


#1.  데이터 불러오기 
train     = pd.read_csv('./data/dacon/bio/train.csv',index_col=0, header=0)
x_predict = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
y_predict = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)
print(x_predict.info())
print(train.shape)   # 10000 75
print(x_predict.shape)    # 10000 71
print(y_predict.shape) # 10000 4

x = train.iloc[:,:71]
print(x.head())
y = train.iloc[:,71:]
print(y.head())

print(x.shape) # 10000, 71
print(y.shape) # 10000, 4

# Nan 값 처리
x = x.fillna(x.mean())
x_predict = x_predict.fillna(x_predict.mean())
print(x.info())
print(y.info())
print(x_predict.info())
print(y_predict.info())

# print(train.isnull().sum()[train.isnull().sum().values > 0])
# print(test.isnull().sum()[test.isnull().sum().values > 0]) 

#. 데이터 저장
np.save('./data/dacon/bio/x.npy', arr=x)
np.save('./data/dacon/bio/y.npy', arr=y)
np.save('./data/dacon/bio/x_predict.npy', arr=x_predict)
np.save('./data/dacon/bio/y_predict.npy', arr=y_predict)



#. 데이터 로드
x         = np.load('./data/dacon/bio/x.npy', allow_pickle='ture')
y         = np.load('./data/dacon/bio/y.npy', allow_pickle='ture')
x_predict = np.load('./data/dacon/bio/x_predict.npy', allow_pickle='ture')
y_predict = np.load('./data/dacon/bio/y_predict.npy', allow_pickle='ture')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False,)
print(x_train.shape) #8000 71
print(y_train.shape) #8000 4
print(x_test.shape)  # 2000, 71
print(y_test.shape)  # 2000, 4
print(x_predict.shape) # 10000 71
print(y_predict.shape) # 10000 71

# print('sum nun : ', x_test.info())


model = DecisionTreeRegressor(max_depth=3)

model.fit(x_train, y_train)

r2 = model.score(x_test, y_test)
print('r2 : ', r2)
# r2 :  -0.015378406881999266
print(model.feature_importances_)


submission = model.predict(x_predict)
print('submission : ', submission.shape)


'''
[0.         0.06741318 0.10128752 0.14593959 0.         0.
 0.0685873  0.         0.05486449 0.         0.         0.07364345
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.07450708 0.         0.
 0.13631894 0.14353884 0.         0.         0.06206984 0.
 0.         0.         0.         0.07182979 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]
 '''


'''
 [0.         0.         0.         0.1597596  0.         0.
 0.         0.         0.         0.15830364 0.11987125 0.17659333
 0.         0.         0.         0.         0.         0.
 0.         0.13685495 0.         0.         0.         0.
 0.14468398 0.10393325 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]
'''