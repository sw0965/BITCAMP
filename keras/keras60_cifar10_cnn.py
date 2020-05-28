from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

# x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
# x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

input1 = Input(shape=(32, 32, 3))
conv1 = Conv2D(64, (2,2), activation='relu', padding='same')(input1)
conv2 = Conv2D(128, (2,2), activation='relu', padding='same')(conv1)
conv3 = Conv2D(128, (2,2), activation='relu')(conv2)
drop = Dropout(0.2)(conv3)
maxp1 = MaxPooling2D(pool_size=2)(drop)

flat = Flatten()(maxp1) 
den1 = Dense(100, activation='relu')(flat)
den2 = Dense(10, activation='softmax')(den1)

model = Model(inputs = input1, outputs = den2)

model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=2, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train,y_train,epochs=1000,batch_size=128,verbose=1 ,validation_split=0.3 ,callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)

y_pre=model.predict(x_test)

y_pre = np.argmax(y_pre,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")



'''1차 cnn튜닝
loss :  1.5311179325103759
acc :  0.7085999846458435
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 1 8 0 6 6 1 4 3 1 0 9 5 7 9 8 5 7 8 6]



2차튜닝 노드갯수 20 -> 100
loss :  2.0885617698669434
acc :  0.6951000094413757
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]


3차 튜닝
loss :  2.10037959690094
acc :  0.6845999956130981
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 8 0 3 6 1 6 3 1 0 9 5 7 9 8 5 3 8 6]
'''
