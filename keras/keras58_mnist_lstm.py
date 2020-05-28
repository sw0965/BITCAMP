import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, LSTM, Dropout
from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

# print(x_train.shape)  # (60000, 28, 28)
# print(x_test.shape)   # (10000, 28, 28)  
# print(y_train.shape)  # (60000, )디멘션 하나
# print(y_test.shape)   # (10000, )


# print(x_train[0].shape)  #(28, 28)

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)
# 모델구성

model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(28, 28)))     
model.add(Dense(784, activation='relu'))
# model.add(Dense(784, activation='relu')) 
# model.add(Dense(784, activation='relu')) 
# model.add(Dense(784, activation='relu')) 

model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))                    

model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=3, mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train,epochs=1000,batch_size=128,verbose=1, callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print('loss : ', loss)
print('acc : ', acc)

'''
loss :  0.04578557377755642
acc :  0.9842099547386169
'''