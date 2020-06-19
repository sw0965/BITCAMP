#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(3))
model.add(Dense(1000000))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, batch_size=1, epochs=50)

y_predict = model.predict(x)
print(y_predict)