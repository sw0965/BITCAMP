import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
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
print(y_train.shape) # (60000, 10) ? 왜 10이 됬지mnist = 10으로 떨어진다.

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
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
model.add(Conv2D(30, (2,2), input_shape=(28,28,1))) 
# model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Conv2D(30, (2,2), padding='same'))   
model.add(Activation('relu'))

model.add(Conv2D(30, (2,2), padding='same'))   
model.add(Activation('relu'))

# model.add(Conv2D(100, (2,2), padding='same'))   
# model.add(Activation('relu'))

model.add(Conv2D(30, (2,2), padding='same'))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))

model.add(Flatten())    
# model.add(Dense(30, activation = 'relu'))    
# model.add(Dense(20, activation = 'relu'))    
model.add(Dense(10, activation='softmax'))    

# model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train,epochs=17,batch_size=128,verbose=1)# callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)
# print(x_test)
# print(y_test)


# y_predict = model.predict(x_test)




# y_pred = np.argmax(y_pred,axis=1) + 1
# print(y_pred)

# print(y_test)
# print(y_predict)


# onehotencording
# 분류때 남자는 1 여자는 2일때 남자 + 남자 = 여자가 아니므로 onehotencording을 사용
# 분류하는 갯수 만큼 shape가 늘어남 

'''
loss :  0.013873947184639681
acc :  0.9972497224807739
'''