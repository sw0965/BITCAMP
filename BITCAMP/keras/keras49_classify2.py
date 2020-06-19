import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils




#. 아웃풋찍을때 변경이 되있어야됌

# 다중분류는 무조건 ont-hot인코딩을 한다.
#1. 데이터
x = np.array(range(1, 11))
y = np.array([1,2,3,4,5,1,2,3,4,5])   
y = np_utils.to_categorical(y)
# y = y[:, 5]
y = y[:, 1:]
# tf.argmax(y, axis=0)

print(y)
print(y.shape) #(10, 6)
#6을 5으로 바꾸기
#y_pred 을 숫자로 바꾸기

# y = y.reshape(y[0], y[-1])
# print(y)




#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
#0인데 1 나올수도 있으니 mertrics에서 accuracy를 쓴다.
model.fit(x,y,epochs=1000,batch_size=16,verbose=2) #, callbacks=[early_stopping])

#4 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

x_pred = np.array([1, 2, 3, 4, 5])
y_pred = model.predict(x_pred)
print(y_pred)



y_pred = np.argmax(y_pred,axis=1) + 1
# y_pred = y_pred.reshape(1, -1)

print(y_pred)


'''
loss :  0.00036975689936298296
acc :  1.0
[[9.9998486e-01 5.3156746e-06 1.8488180e-10 3.6655075e-07 9.4684819e-06]
 [3.3566977e-07 9.9999416e-01 5.5394103e-06 3.1569197e-08 9.7421376e-15]
 [2.4962614e-15 9.0964342e-05 9.9989247e-01 1.6565744e-05 1.8969883e-19]
 [5.0020896e-07 6.1399856e-05 3.0819831e-07 9.9992549e-01 1.2294666e-05]
 [1.5557246e-04 2.4303837e-10 1.6651117e-18 4.4450382e-05 9.9980003e-01]]
[1 2 3 4 5]
'''