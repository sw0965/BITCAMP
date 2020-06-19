from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN



#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])   
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = array([50, 60, 70])

x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape :', x.reshape) 


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(8, activation='relu', input_shape=(3, 1)))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# simple 80

model.summary()


#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode='auto') 
model.fit(x, y, epochs=6945, batch_size=13, verbose=2)# callbacks=[early_stopping])    6945 9538 4433 8168 5878
x_predict = x_predict.reshape(1, 3, 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
