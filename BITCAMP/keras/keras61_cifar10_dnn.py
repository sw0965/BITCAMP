import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
# print('y_train[0] : ', y_train[0])
'''
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
print(x_train)
print(x_test)
'''
'''
x_train = x_train.reshape(50000, 32*32*3).astype('float32')/255
x_test = x_test.reshape(10000, 32*32*3).astype('float32')/255

input1 = Input(shape=(32*32*3,))
den1 = Dense(100, activation='relu')(input1) 
den2 = Dense(300,activation='relu')(den1)   
# den3 = Dense(500,activation='relu')(den2)   
# den4 = Dense(1000,activation='relu')(den3)   
den3 = Dense(300,activation='relu')(den2)
den4 = Dense(100,activation='relu')(den3)
den5 = Dense(30,activation='relu')(den4)
# drop = Dropout(0.5)(den3)
den6 = Dense(10, activation='softmax')(den5)

model = Model(inputs = input1, outputs = den6)

# model.save('./model/sample/cifar10/model_cifar.h5')

model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/cifar10-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train,y_train,epochs=100000,batch_size=128,validation_split=0.4,verbose=1, callbacks=[early_stopping, checkpoint])

# model.save_weights('./model/sample/cifar10/weight_cifar.h5')
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)

y_pre=model.predict(x_test)

y_pre = np.argmax(y_pre,axis=1)
y_test = np.argmax(y_test,axis=1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # 에코가x라서 x값은 안넣었는데 만약 들어가면 이런식으로 plt.plot(x, hist.history['loss'])
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') #label 레전드안에 안써주고 라벨을 따로 붙혀줄수있다
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('loss')        # 제목
plt.ylabel('loss')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(loc = 'upper left')  #plot 순서에따라 맞춰서 기입   #upperright는 legend 위치 명시


plt.subplot(2, 1, 2) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('acc')        # 제목
plt.ylabel('acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['acc', 'val_acc'])

plt.show()
'''
'''
loss :  3.9743302803039553
acc :  0.45579999685287476
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 0 8 4 6 0 6 4 9 0 9 5 7 9 8 1 2 8 4]


loss :  3.932445011138916
acc :  0.4413999915122986

'''