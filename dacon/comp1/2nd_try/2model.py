# src dst 분할 input
import pandas as pd
import numpy as np
from keras.models import Input, Model
from keras.layers import Dense
from sklearn.multioutput import MultiOutputRegressor
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score

train     = np.load('./DACON/comp1/1st_try/data/train.npy', allow_pickle='ture')
x_predict = np.load('./DACON/comp1/1st_try/data/test.npy', allow_pickle='ture')
y_predict = np.load('./DACON/comp1/1st_try/data/y_predict.npy', allow_pickle='ture')
print(train.shape)  #(10000, 75)
print(x_predict.shape) #(10000, 71)





src_x_predict = x_predict[:,1:36]
dst_x_predict = x_predict[:,36:]
print(src_x_predict.shape)
print(dst_x_predict.shape)

rho_x = train[:,0]
src_x = train[:,1:36]
dst_x = train[:,36:71]
y = train[:,71:76]

print(rho_x.shape)  #(10000,)
print(src_x.shape) #(10000, 35)
print(dst_x.shape) #(10000, 35)
print(y.shape)     #(10000, 4)


src_x_train, src_x_test, dst_x_train, dst_x_test, y_train, y_test = train_test_split(src_x, dst_x, y, random_state=44, shuffle=True, test_size=0.2)

print(src_x_train.shape) #(8000, 35)
print(src_x_test.shape)  #(2000, 35)

print(dst_x_train.shape) #(8000, 35)
print(dst_x_test.shape)  #(2000, 35)

print(y_train.shape)
print(y_test.shape)

scaler = RobustScaler()
scaler.fit(src_x_train)
src_x_train = scaler.transform(src_x_train)
src_x_test = scaler.transform(src_x_test)
dst_x_train = scaler.transform(dst_x_train)
dst_x_test = scaler.transform(dst_x_test)

src_x_predict = scaler.transform(src_x_predict)
dst_x_predict = scaler.transform(dst_x_predict)


input1 = Input(shape=(35, ))
dense1_1 = Dense(222)(input1)
dense1_2 = Dense(222)(dense1_1)
dense1_3 = Dense(222)(dense1_2)
dense1_4 = Dense(222)(dense1_3)
dense1_5 = Dense(111)(dense1_4)
dense1_6 = Dense(100)(dense1_5)
dense1_7 = Dense(100)(dense1_6)
dense1_8 = Dense(50)(dense1_7)
dense1_9 = Dense(30)(dense1_8)
dense1_10 = Dense(10)(dense1_9)


input2 = Input(shape=(35, ))
dense2_1 = Dense(222)(input2)
dense2_2 = Dense(333)(dense2_1)
dense2_3 = Dense(333)(dense2_2)
dense2_4 = Dense(444)(dense2_3)
dense2_5 = Dense(111)(dense2_4)
dense2_6 = Dense(150)(dense2_5)
dense2_7 = Dense(100)(dense2_6)
dense2_8 = Dense(80)(dense2_7)
dense2_9 = Dense(30)(dense2_8)
dense2_10 = Dense(10)(dense2_9)


merge1 = concatenate([dense1_10, dense2_10])

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

###### output 모델 구성 ######
#첫번째 아웃풋
output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(4)(output1_2)


model = Model(inputs = [input1,input2], outputs = output1_3) 

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) 
es = EarlyStopping(monitor='loss', patience=5, mode='auto') 
model.fit([src_x_train,dst_x_train],
          y_train, epochs=10000, batch_size=64, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss, mse = model.evaluate([src_x_test,dst_x_test], y_test, batch_size=64) 
print("loss : ", loss)
print("mse : ", mse)


# loss :  5.545769432067871
# mse :  5.545769691467285

# stand
# loss :  5.493505867004394
# mse :  5.493505954742432

# min
# loss :  5.482399345397949
# mse :  5.482398986816406

# ro
# loss :  5.477650356292725
# mse :  5.477651596069336

# sub = model.predict([src_x_predict,dst_x_predict])

# a = np.arange(10000,20000)
# submission = pd.DataFrame(sub, a)
# submission.to_csv('./DACON/dacon_sub_csv/bio1.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')
