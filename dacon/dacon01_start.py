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
test       = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
submission = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)

# print('train.shape : ', train.shape)            #(10000, 75)
# print('test.shape : ', test.shape)              #(10000, 71)  = x_predict
# print('submission.shape : ', submission.shape)  #(10000, 4)   = y_predict


############### 데이터 보관 ###############

train = train.interpolate()  # 보관법 / 선형보관(선형일 가능성이 높아서 선형보관 사용)
test = test.interpolate()  # 보관법 / 선형보관(선형일 가능성이 높아서 선형보관 사용)

############### numpy 전환 ###############

train      = train.values
test       = test.values
submission = submission.values

############### 슬라이싱 ###############

x = train[:,:71]
y = train[:,71:]

############### 트레인 테스트 분리 ###############

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size = 0.2)
print(x_train.shape) #(8000, 71)
print(x_test.shape)  #(8000, 4)
print(y_train.shape) #(2000, 71)
print(y_test.shape)  #(2000, 4)

scaler = StandardScaler()
scaler.fit(x_train)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)


#. 모델구성

inputs = Input(shape=(71,))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(356, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(712, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(356, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4)(x)

model = Model(inputs=inputs, outputs=outputs)

# model.summary()

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10)
loss, mae = model.evaluate(x_test, y_test) 
print("loss : ", loss)
print('mae : ', mae)

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
r2 = r2_score(x_test,y_test)
print("r2 : ",r2)


submission = model.predict(test)
print(submission.shape) #(2000, 4)

print(submission)

loss, mae = model.evaluate(test, submission) 
print("loss : ", loss)
print('mae : ', mae)














############### pipe라인 ###############
# parameters = { 'n_estimators' : [10, 100],
#            'max_depth' : [6, 8, 10, 12],
#            'min_samples_leaf' : [8, 12, 18],
#            'min_samples_split' : [8, 16, 20]
#             }


# #2. 모델
# pipe = Pipeline([("scaler", MinMaxScaler()), ('ensemble', RandomForestRegressor())])

# model = RandomizedSearchCV(pipe, parameters, cv=5)

# #3. 훈련 
# model.fit(x_train, y_train)

# #4. 평가, 예측
# acc = model.score(x_test, y_test)
# print("최적의 매개 변수 : ", model.best_estimator_)
# print("acc: ", acc)