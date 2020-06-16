import numpy             as np
import pandas            as pd 
import matplotlib.pyplot as plt


from   sklearn.metrics         import mean_squared_error, r2_score
from   sklearn.preprocessing   import StandardScaler
from   sklearn.model_selection import train_test_split
from   keras.models            import Model
from   keras.layers            import Dense, LSTM, Input, Activation
from   keras.callbacks         import EarlyStopping, ModelCheckpoint
from   keras.layers.merge      import concatenate 


samsungE = np.load('./data/samE.npy', allow_pickle=True)
hite     = np.load('./data/hite.npy', allow_pickle=True)
ht     = hite[:-1,:]
ss = samsungE[:-1,:]


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y = split_xy5(hite, 5, 1)
x2, y = split_xy5(samsungE, 5, 1)

# print([x[0,:], "\n", y[0]])
print(x1.shape)
print(x2.shape)
print(y.shape)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.3)

# print(x1_train.shape)
# print(x1_test.shape)
# print(y2_train.shape)
# print(y2_test.shape)

x1_train = np.reshape(x1_train, (x1_train.shape[0],x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0],x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0],x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0],x2_test.shape[1] * x2_test.shape[2]))

scaler1 = StandardScaler()
scaler1.fit(x1_train)   
x1_train = scaler1.transform(x1_train)
x1_test = scaler1.transform(x1_test)

scaler2 = StandardScaler()
scaler2.fit(x2_train)   
x2_train = scaler2.transform(x2_train)
x2_test = scaler2.transform(x2_test)

print(x1_train.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_test.shape)
print(y.shape)

x1_train = x1_train.reshape(150, 5, 5)
x1_test = x1_test.reshape(353, 5, 5)
x2_train = x2_train.reshape(150, 5, 1)
x2_test = x2_test.reshape(353, 5, 1)

# print(x1_test.shape)
# print(x2_test.shape)
# print(y1_test.shape)
# print(y2_test.shape)


input1 = Input(shape=(5, 5))
dense1_1 = LSTM(30, activation='relu')(input1)
dense1_2 = Dense(50)(dense1_1)
dense1_8 = Dense(80)(dense1_2)
dense1_9 = Dense(100)(dense1_8)
dense1_10 = Dense(100)(dense1_9)


input2 = Input(shape=(5, 1))
dense2_1 = LSTM(30, activation='relu')(input2)
dense2_7 = Dense(50)(dense2_1)
dense2_8 = Dense(80)(dense2_7)
dense2_9 = Dense(100)(dense2_8)
dense2_10 = Dense(100)(dense2_9)

# from keras.layers.merge import concatenate #단순병합
merge1 = concatenate([dense1_10, dense2_10])

middle1 = Dense(30, activation='relu')(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

###### output 모델 구성 ######

output1 = Dense(30, activation='relu')(middle1) #middle 위에 합쳤던거 마지막 이름을 가져다 인풋으로 쓴다.
output1_2 = Dense(7)(output1)
output1_3 = Dense(1)(output1_2)

# output2 = Dense(30)(middle1)
# output2_2 = Dense(7)(output2)
# output2_3 = Dense(1)(output2_2)

model = Model(inputs=[input1, input2], outputs=output1_3)

# model.summary()


modelpath = './model/booktest-{epoch:02d}-{val_loss:.4f}.hdf5'
cp        = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es        = EarlyStopping(monitor='loss', patience=8, mode='auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) 
hist      = model.fit([x1_train, x2_train],y_train, 
                      epochs=361, batch_size=64, validation_split=0.2, 
                      verbose=1) #, callbacks=[es, cp])




loss = model.evaluate([x1_test, x2_test], y_test, batch_size=32)
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=32)

print('lose : ', loss)
print('mse : ', mse)

y_predict = model.predict([x1_test, x2_test])


for i in range(2):
    print('시가 : ', y_test[i], '/ 예측가 : ', y_predict[i])