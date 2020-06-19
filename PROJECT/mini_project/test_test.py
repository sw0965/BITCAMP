import numpy as np
from keras.models import Input, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


x_train   = np.load('./mini_project/npy_data/x_train_test1.npy')
x_test    = np.load('./mini_project/npy_data/x_test_test1.npy')
y_train   = np.load('./mini_project/npy_data/y_train_test1.npy')
y_test    = np.load('./mini_project/npy_data/y_test_test1.npy')
x_predict = np.load('./mini_project/npy_data/x_predict_test1.npy')

# 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same',input_shape=(100, 100, 3), activation= 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(16, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
# model.add(Dense(50))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(4, activation='softmax'))

model.summary()


# 훈련
modelpath = './mini_project/pro_cp_data/2test-{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1, validation_split=0.2, callbacks=[checkpoint,early_stopping])

#모델 저장
model.save('./mini_project/model_save/test_2.h5')


# 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)



loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# 시각화
plt.figure(figsize=(10, 6)) 

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

'''
# print(x_predict)
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict,axis=-1)
print(y_predict)
print('')

# loss :  0.16387484967708588
# acc :  0.96875


for i in y_predict:
    if i == 0:
        print('사과 입니다')
        print('')
    elif i == 1:
        print('바나나 입니다') 
        print('')
    elif i == 2:
        print('포도 입니다.')
        print('')
    elif i == 3:
        print('파인애플 입니다.')
        print('')
'''