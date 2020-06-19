import numpy as np
import matplotlib.pyplot as plt
from   keras.datasets  import cifar100
from   keras.layers    import Conv2D, Dropout, Input, MaxPooling2D, Flatten, Dense
from   keras.models    import Sequential, Model
from   keras.utils     import np_utils
from   keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print('x_train :',x_train.shape)
print('x_test :',x_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)

#1. 데이터 전처리
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)

# print('x_train :',x_train.shape)
# print('x_test :',x_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)

#2. 데이터 정규화
x_train = x_train/255
x_test  = x_test/255

print('x_train :',x_train.shape)
print('x_test :',x_test.shape)

#.3 모델구성

input1  = Input(shape=(32,32,3))
conv1   = Conv2D(20, (3, 3), activation='relu', strides=2, padding='same')(input1)


output1  = Flatten()                       (conv1)
dens2    = Dense(32,   activation='relu')  (output1)
dens3    = Dense(64,   activation='relu')  (dens2)
dens4    = Dense(128,  activation='relu')  (dens3)
dens5    = Dense(256,  activation='relu')  (dens4)
dens6    = Dense(512,  activation='relu')  (dens5)
dens7    = Dense(1024, activation='relu')  (dens6)
drop1    = Dropout(0.5)                    (dens7)
dens8    = Dense(2048, activation='relu')  (drop1)
drop2    = Dropout(0.5)                    (dens8)
dens9    = Dense(1024, activation='relu')  (drop2)
drop3    = Dropout(0.5)                    (dens9)
dens10   = Dense(512,  activation='relu')  (drop3)
dens11   = Dense(256,  activation='relu')  (dens10)
dens12   = Dense(128,  activation='relu')  (dens11)
dens13   = Dense(100,  activation='relu')  (dens12)

model    = Model(inputs=input1, outputs=dens13)

model.summary()

#.4 모델 훈련
tb            = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
es            = EarlyStopping(monitor='acc', patience=5, mode='auto')
modelpath     = './model/{epoch:02d}-{val_acc:.4f}.hdf5'
cp            = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist          = model.fit(x_train, y_train, validation_split=0.5, epochs=1000, batch_size=600, verbose=1, callbacks=[es,cp,tb])
# D:\Study\study\graph>cd tensorboard --logdir=.(텐서보드 확인 cmd에서)

#.5 평가 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss : ', loss)
print('acc : ',acc)


loss_acc = loss,acc


loss     = hist.history['loss']
acc      = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc  = hist.history['val_acc']

print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc : ', loss_acc)

# 그래프 사이즈
plt.figure(figsize=(10, 6))

# loss 그래프
plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'],     marker='.', c='red',  label='loss')  
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 
plt.grid() 
plt.title('loss')       
plt.ylabel('loss')       
plt.xlabel('epoch')         
plt.legend(loc = 'upper right')

# acc 그래프
plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'],     marker='.', c='red',  label='acc')  
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc') 
plt.grid() 
plt.title('acc')     
plt.ylabel('acc')        
plt.xlabel('epoch')           
plt.legend(loc = 'upper right')  
plt.show()


y_pre  = model.predict(x_test)

y_pre  = np.argmax(y_pre,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")
