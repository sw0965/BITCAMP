from numpy import array
from keras.models import Sequential
from keras.layers import Dense, GRU

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])   # -> 4행3열
y = array([4,5,6,7])

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(GRU(8, input_length=3, input_dim=1))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# GRU = 240
model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=7, mode='auto') 
model.fit(x, y, epochs=7736, batch_size=4, verbose=1)# callbacks=[early_stopping])

x_predict = array([5, 6, 7])
x_predict = x_predict.reshape(1, 3, 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
