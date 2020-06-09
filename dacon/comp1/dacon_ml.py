import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
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
test       = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
submission = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)

# print('train.shape : ', train.shape)            #(10000, 75)
# print('test.shape : ', test.shape)              #(10000, 71)  = x_predict
# print('submission.shape : ', submission.shape)  #(10000, 4)   = y_predict


############### 데이터 보관 ###############

train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 
train = train.interpolate() 
test = test.interpolate()
print(train.info())
print(test.info())

############### numpy 전환 ###############

train      = train.values
test       = test.values
submission = submission.values

############### 슬라이싱 ###############

x = train[:,:71]
y = train[:,71:]

############### 트레인 테스트 분리 ###############

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size = 0.2, random_state=44)
print(x_train.shape) #(8000, 71)
print(x_test.shape)  #(8000, 4)
print(y_train.shape) #(2000, 71)
print(y_test.shape)  #(2000, 4)

scaler = StandardScaler()
scaler.fit(x_train)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)




parameters =[{'bootstrap': [True],
 'criterion': ['mae'],
 'max_depth': [None],
 'max_features': ['auto'],
 'max_leaf_nodes': [None],
 'min_impurity_decrease': [0.0],
 'min_impurity_split': [None],
 'min_samples_leaf': [1],
 'min_samples_split': [2],
 'min_weight_fraction_leaf': [0.0],
 'n_estimators': [10],
 'n_jobs': [1],
 'oob_score': [False],
 'random_state': [42],
 'verbose': [0],
 'warm_start': [False]}
]

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold)# cv= 5라고 써도됌 

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)



y_pred = model.predict(x_test)
print("최종 정답률 : ", r2_score(y_test, y_pred))


'''
최적의 매개변수 :  RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=10, n_jobs=1, oob_score=False,
                      random_state=42, verbose=0, warm_start=False)
'''                    