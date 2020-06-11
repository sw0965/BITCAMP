import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor

#. 데이터 로드
x         = np.load('./data/dacon/bio/x.npy', allow_pickle='ture')
y         = np.load('./data/dacon/bio/y.npy', allow_pickle='ture')
x_predict = np.load('./data/dacon/bio/x_predict.npy', allow_pickle='ture')
y_predict = np.load('./data/dacon/bio/y_predict.npy', allow_pickle='ture')

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False,)
# print(x_train.shape) #8000 71
# print(y_train.shape) #8000 4
# print(x_test.shape)  # 2000, 71
# print(y_test.shape)  # 2000, 4
# print(x_predict.shape) # 10000 71
# print(y_predict.shape) # 10000 71

# print('sum nun : ', x_test.info())


model = XGBRFRegressor()

def model_fit(y) :
    model.fit(x,y)               
    data = model.predict(x_predict)
    df = pd.Series(data)
    return df
# print(y[:,0].shape) # (8000, )


hhb = model_fit(y[:, 0])
hbo2 = model_fit(y[:, 1])
ca = model_fit(y[:, 2])
na = model_fit(y[:, 3])

df = pd.DataFrame([hhb,hbo2,ca,na])
print(df) # (4, 10000)
df = df.transpose()
print(df) # (10000, 4)



# a = np.arange(10000,20000)
df.index =[i for i in range(10000,20000,1)]
# y_pred = pd.DataFrame(df,a)
# y_pred.to_csv('./comp1_submission1.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')
df.to_csv('./comp1_submission1.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')