from numpy import array
# import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


# import numpy as np
# x = np.array 이건 원하는데로 차이점은 np.array or 그냥 array

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])   # -> 4행3열
y = array([4,5,6,7])
y2 = array([[4,5,6,7]])
y3 = array([[4], [5], [6], [7]])

# 스칼라 = 1하나가 스칼라, 2하나가 스칼라 이렇게 따로따로
# 벡터 = 스칼라가 이어진거 y = 벡터(괄호)가 1개 스칼라 4개짜리 



print('x.shape : ', x.shape)    # (4, 3)
print('y.shape : ', y.shape)    # (4, )  
print('y2.shape : ', y2.shape)  # (1, 4)  -> 스칼라가 4개라는 뜻
print('y3.shape : ', y3.shape)  # (4, 1)  


# x = x.reshape(4, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(8, activation='relu', input_shape=(3, 1)))     
#위 x 값 (4, 3, 1(이 1은 1개씩 자르겠다) 에서 행 무시하고 열 3과 1이 남으므로 (3, 1))
#LSTM 에서 원하는 모양은 열과 몇개씩 자를건지.
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


model.summary()

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
#과제: 05/20 parameter 480의미 찾기