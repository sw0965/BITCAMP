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
print(test.isnull().sum()[test.isnull().sum().values > 0])


############### 데이터 보관 ###############

train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 
print(test.isnull().sum()[test.isnull().sum().values > 0])

train = train.interpolate() 
test = test.interpolate()
print(test.isnull().sum()[test.isnull().sum().values > 0])

# print(test.info())

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
scaler.fit(x)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)


#. 모델구성

inputs = Input(shape=(71,))
x = Dense(128)(inputs)
x = Dropout(0.5)(x)
x = Dense(356)(x)
x = Dropout(0.5)(x)
# x = Dense(712)(x)
x = Dropout(0.5)(x)
# x = Dense(356)(x)
x = Dropout(0.5)(x)
x = Dense(128)(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4)(x)

model = Model(inputs=inputs, outputs=outputs)

# model.summary()
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=66, batch_size=8, callbacks=[early_stopping])
loss, mae = model.evaluate(x_test, y_test) 
print("loss : ", loss)
print('mae : ', mae)

submission = model.predict(test)
print(submission.shape) #(10000, 4)

print(submission)

print(type(submission))

# submission = pd.DataFrame(submission)
print(type(submission))

a = np.arange(10000,20000)
y_pred = pd.DataFrame(submission,a)
y_pred.to_csv('./sample_submission2.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')



# loss, mae = model.evaluate(test, submission) 
# print("loss : ", loss)
# print('mae : ', mae)



# 125
# loss :  1.7163434734344483
# mae :  1.7163432836532593

# 115
# loss :  1.7199207229614257
# mae :  1.7199207544326782

# 114
# loss :  1.7177422847747803
# mae :  1.7177422046661377
