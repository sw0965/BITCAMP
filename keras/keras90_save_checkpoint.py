import numpy as np 
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

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)
# 모델구성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
model = Sequential()
model.add(Conv2D(10, (3,3), padding='same', input_shape=(28,28,1))) 
model.add(Conv2D(100, (3,3), padding='same'))       
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(300, (3,3), padding='same'))   
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(500, (3,3), padding='same'))   
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(100, (3,3), padding='same'))   
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())    
model.add(Dense(10, activation='softmax'))                    
# model.add(Conv2D(5, (2,2), strides=2))              
# model.add(Conv2D(5, (2,2), strides=2, padding='same')) 
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())  # 쫙 피는거 (3,3) 3장있으면 가로로 이어서 9개
# model.add(Dense(1))

# model.summary()

# model.save('./model/model_test01.h5')

#3. 훈련
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=5)

hist = model.fit(x_train,y_train,epochs=50,batch_size=32,validation_split=0.2,verbose=1, callbacks=[early_stopping, checkpoint])
model.save('./model/model_test01.h5')



#4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=1)
loss, acc = loss_acc
print('loss : ', loss)
print('acc : ', acc)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc : ', loss_acc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # 에코가x라서 x값은 안넣었는데 만약 들어가면 이런식으로 plt.plot(x, hist.history['loss'])
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') #label 레전드안에 안써주고 라벨을 따로 붙혀줄수있다
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('loss')        # 제목
plt.ylabel('loss')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(loc = 'upper right')  #plot 순서에따라 맞춰서 기입   #upperright는 legend 위치 명시


plt.subplot(2, 1, 2) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('acc')        # 제목
plt.ylabel('acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['acc', 'val_acc'])

plt.show()


# print(x_test)
# print(y_test)


y_predict = model.predict(x_test)




y_pred = np.argmax(y_predict,axis=1) + 1
y_test = np.argmax(y_test,axis=-1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pred[0:20]}")


# print(y_test)
# print(y_predict)

'''
loss :  0.10222576545828196
acc :  0.9871000051498413
'''

