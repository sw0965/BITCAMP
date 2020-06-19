import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1, 31))
size = 3

# print(a.shape)

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq [i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape)

x = dataset[:20, 0:2]
print(x.shape)
y = dataset[:20, 2]
print(y)
x_predict = dataset[20:, 0:2]
print(x_predict.shape)


x = np.reshape(x, (20, 2, 1))

x_predict = x_predict.reshape(8, 2, 1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=False, train_size=0.7)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(2, 1)))     
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_train,y_train)

y_predict = model.predict(x_predict)
print(x_predict)
print(y_predict)