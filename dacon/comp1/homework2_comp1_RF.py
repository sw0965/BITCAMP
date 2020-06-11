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
print(y_predict.shape) # 10000 4

# print('sum nun : ', x_test.info())

model = RandomForestRegressor() 

#max_features :기본값 쓰기
#n_estimators : 클수록 좋다 단점 메모리 많이 차지 기본값 100
#n_jobs=-1    : 병렬처리

# 훈련 테스트
model.fit(x_train, y_train)

# 훈련결과
r2 = model.score(x_test, y_test)
print('r2 : ', r2)
# r2 :  -0.04078988552505847
print(model.feature_importances_)

# 프레딕트
submission = model.predict(x_predict)
print('sub : ', submission.shape) # 10000,4



'''
[0.02012541 0.03055724 0.02982249 0.0300189  0.0292532  0.02839962
 0.03011524 0.02783363 0.02917114 0.02894184 0.0286216  0.02901547
 0.03038343 0.02928926 0.0285115  0.02929076 0.02758773 0.02706113
 0.02608538 0.02503318 0.02458885 0.0259717  0.02566554 0.0265261
 0.02747307 0.02860991 0.02645424 0.028017   0.02731935 0.02744422
 0.02655061 0.02633422 0.02656334 0.02807202 0.02883072 0.03046097
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.        ]
 '''




# def plot_feature_importances_bio(model):
#     n_features = x.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), pd[x.feature_names])
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)

# plot_feature_importances_bio(model)
# plt.show()