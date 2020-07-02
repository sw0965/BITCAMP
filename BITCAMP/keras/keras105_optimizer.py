#1. 데이터 
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam, Adamax   # 경사 하강법베이스로 둔 것중 하나이다.
optimizer = Adam(lr=0.001)          #loss = 0.04252109304070473,  pred =  3.467478   dif btw pred = 0.032522
# optimizer = RMSprop(lr=0.001)     #loss = 0.013454725965857506, pred =  3.4440575  dif btw pred = 0.0559425
# optimizer = Nadam(lr=0.001)       #loss = 0.027311410754919052, pred =  3.4247665  dif btw pred = 0.0752335
# optimizer = SGD(lr=0.001)         #loss = 0.033921096473932266, pred =  3.4173672   
# optimizer = Adamax(lr=0.001)      #loss = 0.15930819511413574,  pred =  2.9921505
# optimizer = Adagrad(lr=0.001)     #loss = 3.6516146659851074,   pred =  1.0311925   
# optimizer = Adadelta(lr=0.001)    #loss = 20.337982177734375,   pred = -2.2637787   

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print('loss :', loss)

pred = model.predict([3.5])
print(pred)