import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

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

x_train = x_train.reshape(50000, 32*32*3).astype('float32')/255
x_test = x_test.reshape(10000, 32*32*3).astype('float32')/255

input1 = Input(shape=(32*32*3,))
den1 = Dense(100, activation='relu')(input1) 
den2 = Dense(300,activation='relu')(den1)   
den3 = Dense(500,activation='relu')(den2)   
den4 = Dense(1000,activation='relu')(den3)   
den5 = Dense(300,activation='relu')(den4)
den6 = Dense(100,activation='relu')(den5)
den7 = Dense(30,activation='relu')(den6)
# drop = Dropout(0.5)(den3)
den8 = Dense(10, activation='softmax')(den7)

model = Model(inputs = input1, outputs = den8)

model.save('./model/sample/cifar10/model_cifar.h5')

model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './model/sample/cifar10/cifar10-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping = EarlyStopping(monitor='acc', patience=3, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train,epochs=100000,batch_size=128,validation_split=0.2,verbose=1, callbacks=[early_stopping, checkpoint])

model.save_weights('./model/sample/cifar10/weight_cifar.h5')
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)

y_pre=model.predict(x_test)

y_pre = np.argmax(y_pre,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")


'''
loss :  3.9743302803039553
acc :  0.45579999685287476
y_test[0:20]:[3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 7 8 6]
y_pre[0:20]:[3 8 0 8 4 6 0 6 4 9 0 9 5 7 9 8 1 2 8 4]

'''