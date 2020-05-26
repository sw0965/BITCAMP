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
tf.argmax(y, axis=0)
print(y)
print(y.shape) #(10, 6)
#6을 5으로 바꾸기
#y_pred 을 숫자로 바꾸기

# y = y.reshape(y[0], y[-1])
# print(y)



'''
#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
#0인데 1 나올수도 있으니 mertrics에서 accuracy를 쓴다.
model.fit(x,y,epochs=100,batch_size=1,verbose=2) #, callbacks=[early_stopping])

#4 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)
print(y_pred)
'''