from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN



#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])   # -> 4행3열
y = array([4,5,6,7])




print('x.shape : ', x.shape)    # (4, 3)
print('y.shape : ', y.shape)    # (4, )  


x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

# sim = 80 
#2. 모델구성
model = Sequential()
model.add(SimpleRNN(8, activation='relu', input_shape=(3, 1)))     
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
'''

#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') 
model.fit(x, y, epochs=249, batch_size=1, verbose=1)# callbacks=[early_stopping])

x_input = array([5, 6, 7])
x_input = x_input.reshape(1, 3, 1)

print(x_input)

yhat = model.predict(x_input)
print(yhat)

#LSTM을 가지고 뽑긴 데이터가 적어서 값이 잘 안나온다. 그러므로 튜닝

#----------------------
#과제: 05/21
#성능비교와 연산갯수 차이
'''