import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

# 주어진 데이터
train_features = pd.read_csv('./data/dacon/KAERI/train_features.csv', header=0, index_col=0)
train_target = pd.read_csv('./data/dacon/KAERI/train_target.csv', header=0, index_col=0)
#적용 데이터
test_features = pd.read_csv('./data/dacon/KAERI/test_features.csv', header=0, index_col=0)
test_target = pd.read_csv('./data/dacon/KAERI/sample_submission.csv', header=0, index_col=0)


x = train_features          # time s1 s2 s3 s4
y = train_target            # x  y  m   v
x_predict = test_features   # time s1 s2 s3 s4
y_predict = test_target     #

print(x.shape)         #1050000, 5
print(y.shape)         #2800, 4
print(x_predict.shape) #262500,5
# print(x.shape)
x = x.values        #array
y = y.values        #array
x_predict=x_predict.values  #array


x_predict = x_predict[:,1:]
x_predict = x_predict.reshape(700,375,4)


x = x[:,1:]
print(x.shape) # 1050000, 4

# scaler = MinMaxScaler()


x = x.reshape(2800,375,4)

print(x.shape) #2800, 375, 4
print(y.shape) # 2800, 4

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)  # 2240, 375, 4
print(x_test.shape)   # 560, 375, 4
print(y_train.shape)  # 2240, 4
print(y_test.shape)   # 560, 4

x_train = x_train.reshape(840000, 4)
x_test = x_test.reshape(210000, 4)
x_predict = x_predict.reshape(262500, 4)
# x_train = x_train.reshape(840000, 4)
# x_train = x_train.reshape(840000, 4)

scaler = StandardScaler()
scaler.fit(x_train)   #전처리에서 fit 실행하다. 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(2240, 375, 4)
x_test = x_test.reshape(560, 375, 4)
x_predict = x_predict.reshape(700,375,4)


model = Sequential()
model.add(Conv1D(64, 2, input_shape=(375, 4)))
model.add(MaxPooling1D())
model.add(Flatten())
# model.add(acitvation = 'leakyReLU')
# model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(356))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))

# model.summary()


es = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss= 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10000, callbacks=[es])

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

submission = model.predict(x_predict)
print(submission)
