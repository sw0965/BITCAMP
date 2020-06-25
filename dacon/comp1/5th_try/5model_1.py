# 1st data 사용
# 자르지 않고 rho,src,dst 한번에 multi liner regresso 사용

import numpy as np
import pandas as pd
from sklearn.model_selection  import  train_test_split
from xgboost import XGBRegressor, plot_importance ,XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from keras.layers import Dense
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt


train     = np.load('./DACON//comp1/1st_try/data/train.npy', allow_pickle='ture')
x_predict = np.load('./DACON//comp1/1st_try/data/test.npy', allow_pickle='ture')
y_predict = np.load('./DACON//comp1/1st_try/data/y_predict.npy', allow_pickle='ture')

# print(train)      # (10000, 75)
# print(x_predict)      #  (10000, 71)
# print(y_predict)  # (10000, 4)

print(x_predict.shape)        
# print(y_predict.shape)   

# rho = train[:,:1]
# print(rho)
# src = train[:,1:36]
# dst = train[:,36:71]
y = train[:,71:]
x = train[:,:71]

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=55, test_size=0.2)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

model = LinearRegression()



model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(' 점수 : ', score)

y_pred = model.predict(x_predict)

a = np.arange(10000,20000)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('./DACON/dacon_sub_csv/comp1/5model.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')