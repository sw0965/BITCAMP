import numpy as np
import pandas as pd 

samsungE = np.load('./data/samE.npy', allow_pickle=True)
hite     = np.load('./data/hite.npy', allow_pickle=True)


# print(samsungE)
print(hite)
# print(samsungE.shape)
# print(hite.shape)

x = hite[:-1,:]
y = samsungE[:-1,:]
# print('slice : ', hite)
# print('slice : ', samsungE)
print('b4 std : ', x)

from sklearn.preprocessing import StandardScaler, Normalizer
scaler = Normalizer()
scaler.fit(x)   
x = scaler.transform(x)

x1 = x[:254]
x2 = x[254:]
# y1 = y[:254]
# y2 = y[254:]



# print('b4 pca : ', x)
'''
b4 pca :  [[-0.1792711  -0.20289655 -0.13302698 -0.15678522 -0.91870954]
 [-0.17014953 -0.21173478 -0.19848207 -0.24732392 -0.37736274]
 [-0.24312206 -0.27360239 -0.2078328  -0.23827005 -0.74127311]
 ...
 [ 2.52071255  2.39554324  2.51322888  2.42356765  0.89348947]
 [ 2.46598315  2.47508732  2.5880347   2.45978313  1.01322057]
 [ 2.48422629  2.82861654  2.60673616  2.95774596  4.55648568]]
'''
'''
from   sklearn.decomposition    import PCA
pca = PCA(n_components=4)
pca.fit(x)
x = pca.transform(x)
print('pca : ', x)
print('x.shape :', x.shape) #(506, 4)
'''
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(x)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
print(x_train.shape)  #406 5 
print(x_test.shape)   #102 5
print(y_train.shape)  #406 1 
print(y_test.shape)   #102 1
'''
'''
x1_train = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
print('x1.reshape :', x1.shape) 
print('x2.reshape :', x2.shape) 


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input


# 모델 구성
input1 = Input(shape=(5, 1))
dense1_1 = LSTM(1, name='lstm1')(input1)
dense1_2 = Dense(7, name='dense1')(dense1_1)

input2 = Input(shape=(5, 1))
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
'''
'''
# model = Sequential()
# model.add(Dense(50, input_shape=(5,)))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1))

model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath     = './model/test-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss = 'mse' , optimizer='adam', metrics=['mse'],) 
hist = model.fit([x1_train, y1_train],[x2_train, y2_test], epochs=1000, batch_size=32, validation_split=0.4, verbose=1, callbacks=[es, cp])



loss, mse = model.evaluate([x1_train, y1_train],[x2_train, y2_test], batch_size=1)
print('loss : ', loss)
print('mse : ', mse)


y_pred = model.predict([x1_test, x2_test])
# print(y_pred)

from   sklearn.metrics          import mean_squared_error, r2_score
#RMSE 구하기 #낮을수록 좋다
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test,y1_predict)
r2_2 = r2_score(y2_test,y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1+r2_2)/2)
'''
'''
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  # 에코가x라서 x값은 안넣었는데 만약 들어가면 이런식으로 plt.plot(x, hist.history['loss'])
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') #label 레전드안에 안써주고 라벨을 따로 붙혀줄수있다
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('loss')        # 제목
plt.ylabel('loss')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(loc = 'upper right')  #plot 순서에따라 맞춰서 기입   #upperright는 legend 위치 명시


plt.subplot(2, 1, 2) #두개의 그림을 그림  (2행 1열에 첫번째 그림을 그리겠다)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() # 모눈종이처럼 가로세로 줄이 그어짐
plt.title('acc')        # 제목
plt.ylabel('acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['acc', 'val_acc'])

plt.show()
'''
