import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )


print(x_train[0].shape)  #(28, 28)
# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# # plt.show()

#0 부터 9까지 분류 onehotencording
#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('yshape :', y_train.shape) # (60000, 10) ? 왜 10이 됬지mnist = 10으로 떨어진다.

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
# reshape로 왜 4차원으로 만들었나?? cnn모델은 4차원이기 때문에
# 안에 들어가는 숫자는 정수형태 0 부터 255까지
# (x에 완전 진한검정 255 x엔 255까지 들어가있다)
#minmax 는 0 부터 1은 실수라 'float32'(실수로 만드는거 같음 찾아봐야됌) 처리 하는거
# 마지막 나누기 255는 0부터 1까지 사이를 넣기위해 정규화 시키기 위해
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
model.add(Dense(784, input_shape=(28*28,)))
# model.add(Flatten(input_shape=(28,28)))    
model.add(Activation('relu'))

model.add(Dense(784))
# model.add(Dropout(0.3))
model.add(Activation('relu'))


model.add(Dense(784))   
model.add(Activation('relu'))
model.add(Dropout(0.5))

# model.add(Dense(784))   
# model.add(Activation('relu'))

# model.add(Conv2D(100, (2,2), padding='same'))   
# model.add(Activation('relu'))

model.add(Dense(784))   
model.add(Activation('relu'))
# model.add(Dropout(0.3))
model.add(Dropout(0.3))


# model.add(Flatten())    
model.add(Dense(10, activation='softmax'))    

# model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=3, mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train,epochs=28,batch_size=128,verbose=1)#, callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print('loss : ', loss)
print('acc : ', acc)


'''
loss :  0.023212459191205636
acc :  0.9961099624633789
'''


