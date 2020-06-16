import numpy as np
from keras.models import Input, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#데이터 로드
x         = np.load('./data/mini_project/x_data.npy')
y         = np.load('./data/mini_project/y_data.npy')
x_predict = np.load('./data/mini_project/x_predict.npy')
print(x.shape)         #(160, 100, 100, 3)
print(y.shape)         #(160, 4)
print(x_predict.shape) #(20, 100, 100, 3)


# train test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape) #(128, 100, 100, 3)
print(x_test.shape)  #(32, 100, 100, 3)
print(y_train.shape) #(128 , 4)
print(y_test.shape)  #(32 , 4)


# 스케일러 위해 리쉐이프
x_train = x_train.reshape(128, 30000)
x_test = x_test.reshape(32, 30000)
x_predict = x_predict.reshape(20, 30000)

x_train = x_train/255
x_test = x_test/255

# 스케일러
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)


# 모델링을 위해 다시 리쉐이프
x_train = x_train.reshape(128, 100, 100, 3)
x_test = x_test.reshape(32, 100, 100, 3)
x_predict = x_predict.reshape(20, 100, 100, 3)


# 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(100, 100, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(16))
model.add(Dense(40000))
model.add(Dense(7))
model.add(Dense(5000))
model.add(Dense(6))
model.add(Dense(500))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(4, activation='softmax'))

model.summary()



modelpath = './mini_project/pro_es_data/-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train,y_train,epochs=23,batch_size=32, verbose=1 ,validation_split=0.2 ,callbacks=[checkpoint,early_stopping])


loss, acc = model.evaluate(x_test, y_test,)
print('loss : ', loss)
print('acc : ', acc)


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']




plt.figure(figsize=(10, 6)) #가로세로 길이 그래프설정

plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss') 
plt.ylabel('loss') 
plt.xlabel('epoch')  
plt.legend(loc = 'upper right') 


plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid() 
plt.title('acc')    
plt.ylabel('acc')  
plt.xlabel('epoch') 
plt.legend(['acc', 'val_acc'])

plt.show()