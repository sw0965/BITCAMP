import numpy             as np
import pandas            as pd 
import matplotlib.pyplot as plt

from   sklearn.metrics         import mean_squared_error, r2_score
from   sklearn.model_selection import train_test_split
from   keras.models            import Sequential, Model
from   keras.layers            import Dense, LSTM, Input, Activation
from   keras.callbacks         import EarlyStopping, ModelCheckpoint
from   keras.layers.merge      import concatenate 


#데이터
samsungE = np.load('./data/samE.npy', allow_pickle=True)
hite     = np.load('./data/hite.npy', allow_pickle=True)


#데이터 슬라이싱
x = hite[:-1,:]
y = samsungE[:-1,:]

from sklearn.preprocessing import StandardScaler, Normalizer
scaler = StandardScaler()
scaler.fit(x)   
x = scaler.transform(x)

x1 = x[:254]
x2 = x[254:]
# y1 = y[:254]
# y2 = y[254:]
print(x1.shape)  #254 5
print(x2.shape)  #254 5
print(y.shape)  #254 1
# print(y2.shape)  #254 1
y = y.reshape(254, 2)
#데이터 트레인 테스트 분류
x1_train, x1_test, x2_train, x2_test, y_train, y_test  = train_test_split(x1, x2, y, train_size=0.5)

print(x1_train.shape)  #(127, 5)
print(x1_test.shape)   #(127, 5)
print(x2_train.shape)  #(127, 5)
print(x2_test.shape)   #(127, 5)
print(y_train.shape)  #(127, 1)
print(y_test.shape)   #(127, 1)
# print(y2_train.shape)  #(127, 1)
# print(y2_test.shape)   #(127, 1)

#데이터 리쉐이프
x1_train = x1_train.reshape(127, 5, 1)
x2_train = x2_train.reshape(127, 5, 1)
x1_test  = x1_test.reshape (127, 5, 1)
x2_test  = x2_test.reshape (127, 5, 1)
print('x1.reshape :', x1_train.shape)   #(127, 5, 1)
print('x2.reshape :', x2_train.shape)   #(127, 5, 1)
 



# 모델 구성
input1     = Input(shape=(5, 1))
dense1_1   = LSTM(1, name='lstm1')     (input1)
dense1_2   = Dense(10, name='dense1')  (dense1_1)

input2     = Input(shape=(5, 1))
dense2_1   = LSTM(1, name='dense2')    (input2)
dense2_2   = Dense(10, name='dense3')  (dense2_1)

merge1     = concatenate([dense1_2, dense2_2])
middle1    = Dense(80, name='mid1')    (merge1)
middle1    = Dense(70, name='mide2')   (middle1)

output1    = Dense(50)                 (middle1) 
output1_2  = Dense(30)                 (output1)
activation = Activation('linear')      (output1_2)
output1_3  = Dense(2)                  (activation)


model      = Model(inputs = [input1, input2], outputs = output1_3)

model.summary()

# 컴파일, 핏
modelpath = './model/test-{epoch:02d}-{val_loss:.4f}.hdf5'
cp        = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es        = EarlyStopping(monitor='loss', patience=8, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) 
model.fit([x1_train, x2_train],y_train, 
                      epochs=1000, batch_size=32, validation_split=0.2, 
                      verbose=1, callbacks=[es, cp])



loss = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
mse  = model.evaluate([x1_test, x2_test], y_test, batch_size=1)

print('loss : ', loss)
print('mse : ', mse)


y_predict = model.predict([x1_test, x2_test])


#RMSE 구하기 #낮을수록 좋다.
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE1 = RMSE(y_test, y_predict)
print("RMSE : ", RMSE1)

#R2 구하기 # 1에 근접할수록 좋다.
r2 = r2_score(y_test,y_predict)
print("R2 : ", r2)






for i in range(5):
    print('시가 : ', y_test[i], '/ 예측가 : ', y_predict[i])

