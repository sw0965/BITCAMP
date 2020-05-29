import matplotlib.pyplot as plt
import numpy as np

from   sklearn.datasets         import load_iris
from   sklearn.model_selection  import train_test_split
from   keras.utils              import np_utils
from   keras.models             import Sequential
from   keras.layers             import Dense, Dropout
from   keras.callbacks          import EarlyStopping, ModelCheckpoint, TensorBoard

#1. 데이터
dataset = load_iris()

x       = dataset.data
y       = dataset.target

print('x.shape :', x.shape)
print('y.shape :', y.shape)

# train, test 분류
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 66, shuffle=True, train_size=0.7)

print('x_train.shape : ',x_train.shape)  # (105, 4)
print('x_test.shape : ',x_test.shape)    # (45, 4)
print('y_train.shape : ',y_train.shape)  # (105, )
print('y_test.shape : ',y_test.shape)    # (45, )



#데이터 전처리 1. 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('y.shape :', y_train.shape) # (105, 3)
print(x.dtype)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(105, 4).astype('float32')/255
x_test = x_test.reshape(45, 4).astype('float32')/255

print(x_train.shape)  #(105, 4)
print(x_test.shape)   #(45, 4)
print(y_train.shape)  #(105, 3)
print(y_test.shape)   #(45, 3)


# 모델구성

model = Sequential()
model.add(Dense(4, input_shape=(4,)))
model.add(Dense(8))
model.add(Dense(16))   
model.add(Dense(8))   
model.add(Dense(4))   
model.add(Dense(3, activation='softmax'))    

model.summary()

#.4 모델 훈련
tb            = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
es            = EarlyStopping(monitor='acc', patience=2, mode='auto')
modelpath     = './model/{epoch:02d}-{val_acc:.4f}.hdf5'
cp            = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist          = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=1, callbacks=[es,cp,tb])
# D:\Study\study\graph>cd tensorboard --logdir=.(텐서보드 확인 cmd에서)

#.5 평가 예측
loss,acc = model.evaluate(x_test, y_test, batch_size=32)
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

