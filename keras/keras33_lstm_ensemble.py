# 앙상블 모델로 리모델하시오
# 2개들어가서 하나 나오는거.

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input



#1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])   
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])   

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# print('x1.reshape :', x1.reshape) 
# print('x2.reshape :', x2.reshape) 



#2. 모델구성

input1 = Input(shape=(3, 1))
dense1_1 = LSTM(1, name='lstm1')(input1)
dense1_2 = Dense(7, name='dense1')(dense1_1)

input2 = Input(shape=(3, 1))
dense2_1 = LSTM(1, name='dense2')(input2)
dense2_2 = Dense(7, name='dense3')(dense2_1)

from keras.layers.merge import concatenate #단순병합
merge1 = concatenate([dense1_2, dense2_2])
middle1 = Dense(8, name='mid1')(merge1)
middle1 = Dense(7, name='mide2')(middle1)

output1 = Dense(4, name='ot1')(middle1)
output1_2 = Dense(2, name='ot2')(output1)
output1_3 = Dense(1, name='ot3')(output1_2)


model = Model(inputs = [input1,input2], outputs = output1_3)
model.summary()


#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 
model.fit([x1, x2], y, epochs=11000000, batch_size=16, verbose=2, callbacks=[early_stopping])    #6489 6817 2001 5158

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)


print(x1_predict)
print(x2_predict)


y_predict = model.predict([x1_predict, x2_predict])
# y_predict2 = model.predict(x2_predict)

print(y_predict)


