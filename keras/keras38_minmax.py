
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input


#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],[100,200,300]])   
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict = array([55, 65, 75])

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit(x)   #전처리에서 fit 실행하다. 
x = scaler.transform(x)

print(x)




x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape :', x.reshape) 



#2. 모델구성
# return_sequences=True
input1 = Input(shape=(3, 1)) 
input1_1 = LSTM(10, return_sequences=True, name='lstm1')(input1)
input1_1 = LSTM(10, return_sequences=False, name='lstm2')(input1_1)
input1_1 = Dense(5, name='dense1')(input1_1)
input1_1 = Dense(1, name='dense2')(input1_1)
input1_1 = Dense(2, name= 'dense3')(input1_1)


output1 = Dense(10, name='dense4')(input1_1)
output1_1 = Dense(2, name='dense5')(output1)
output1_1 = Dense(6, name='dense6')(output1_1)
output1_1 = Dense(1, name='dense7')(output1_1)
output1_1 = Dense(1, name='dense8')(output1_1)
output1_1 = Dense(1, name='dense9')(output1_1)


model = Model(inputs = input1, outputs = output1_1)


model.summary()

#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss='mse',optimizer='adam') 
model.fit(x, y, epochs=1, batch_size=16, verbose=2, callbacks=[early_stopping])  #5104
x_predict = x_predict.reshape(1, 3, 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
