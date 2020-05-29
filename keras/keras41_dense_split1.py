# 실습 40번을 카피해서 dense 모델로 만드시오.
# 40번과 41번을 비교하면서 높은 loss를 낮게 튜닝하시오.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11)) 
size = 5                           # time_steps = 4
print("a.shape :", a.shape)

# 실습: lstm 모델을 완성하시오. size 5


def split_x(seq, size):    #size = lstm의 timesteps (열)
    aaa = []
    for i in range(len(seq) - size + 1): #<-이게 행 길이에서 - size + 1 = 열 
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])   #가장중요
    print(type(aaa))
    return np.array(aaa)   #리턴값이 numpy라 밑에 type(dataset)이


dataset = split_x(a, size)
print("============================")
print(dataset)
print(dataset.shape)   #(6, 5) 
print(type(dataset))   #class 'numpy.ndarray' 

x = dataset[:, 0:4]    # : 은 all [모든행이 들어가겠다, (0,1,2,3)] 6, 5 짜리 똔똔해서 모든행을 가져오고 0부터 3(4-1) 까지를 가져오겠다.
y = dataset[:, 4]

print(x.shape) #(6, 4)
print(y.shape) #(6, )




#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=4))     
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))


model.summary()


#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x, y, epochs=123, batch_size=1, verbose=1) #, callbacks=[early_stopping])

loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss : ', loss)
print('mse :', acc)
print('y_predict :', y_predict)