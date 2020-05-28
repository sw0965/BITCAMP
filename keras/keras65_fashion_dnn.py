import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

model = Sequential()
model.add(Dense(784, input_shape=(28*28,)))
model.add(Dense(784))
model.add(Dense(784))   
model.add(Dropout(0.5))
model.add(Dense(784))   
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))


model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=2, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
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
