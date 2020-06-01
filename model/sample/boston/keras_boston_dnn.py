from   sklearn.datasets         import load_boston
from   sklearn.model_selection  import train_test_split
from   sklearn.preprocessing    import StandardScaler
from   sklearn.decomposition    import PCA
from   sklearn.metrics          import mean_squared_error, r2_score
from   keras.utils              import np_utils
from   keras.models             import Sequential
from   keras.layers             import Dense, Dropout
from   keras.callbacks          import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np

# 데이터

dataset = load_boston()
x       = dataset.data
y       = dataset.target
print('x :', x.shape) #(506, 13)
print('y :', y.shape) #(506, )

# 전처리
scaler = StandardScaler()
scaler.fit(x)   
x = scaler.transform(x)

pca = PCA(n_components=10)
pca.fit(x)
x = pca.transform(x)
print('x.shape :', x.shape) #(506, 10)



# train, test 분류
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 66, shuffle=True, train_size=0.8)

print('x_train.shape : ',x_train.shape)  # (354, 10)
print('x_test.shape : ',x_test.shape)    # (152, 10)
print('y_train.shape : ',y_train.shape)  # (354, )
print('y_test.shape : ',y_test.shape)    # (152, )


# 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(10,)))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
# model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
model.save('./model/sample/boston/model_boston.h5')
# 훈련

# early_stopping = EarlyStopping(monitor='acc', patience=2, mode='auto')
# model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) 
# model.fit(x_train,y_train,epochs=1000,batch_size=1,verbose=1 ,validation_split=0.4)# ,callbacks=[early_stopping])

es            = EarlyStopping(monitor='mse', patience=10, mode='auto')
modelpath     = './model/sample/boston/boston-{epoch:02d}-{val_loss:.4f}.hdf5'
cp            = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist          = model.fit(x_train, y_train, validation_split=0.5, epochs=1000, batch_size=16, verbose=1, callbacks=[es,cp])
# D:\Study\study\graph>cd tensorboard --logdir=.(텐서보드 확인 cmd에서)
model.save_weights('./model/sample/boston/weight_boston.h5')




#. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)



y_pred = model.predict(x_test)
# print(f"y_test[0:20]:{y_test[0:20]}")
# print(f"y_pred[0:20]:{y_pred[0:20]}")

print('loss : ', loss)
print('mse : ', mse)
#RMSE 구하기 #낮을수록 좋다
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)


# 튜닝필요

loss     = hist.history['loss']
mse      = hist.history['mse']
val_loss = hist.history['val_loss']
val_mse  = hist.history['val_mse']

# print('mse : ', mse)
# print('val_mse : ', val_mse)
# print('loss, mse: ', loss, mse)

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
plt.plot(hist.history['mse'],     marker='.', c='red',  label='mse')  
plt.plot(hist.history['val_mse'], marker='.', c='blue', label='val_mse') 
plt.grid() 
plt.title('mse')     
plt.ylabel('mse')        
plt.xlabel('epoch')           
plt.legend(loc = 'upper right')  
plt.show()





'''
loss :  11.156186711657364
mse :  11.15618896484375
RMSE :  3.340087889019269
r2 :  0.8649651594926209


loss :  21.893379660749098
mse :  21.89337730407715
RMSE :  4.679036190691256
r2 :  0.7350018420962126


loss :  31.871591618921805
mse :  31.8715877532959
RMSE :  5.645493222285012
r2 :  0.6142252246177748


loss :  21.98970543084464
mse :  21.98969268798828
RMSE :  4.689318412249477
r2 :  0.7338358910271394


loss :  14.876608657411692
mse :  14.876612663269043
RMSE :  3.8570208123908083
r2 :  0.8199330489579864


loss :  64.23061387919772
mse :  64.2305908203125
RMSE :  8.014400594377806
r2 :  0.22255058790706073

loss :  21.642169174726813
mse :  21.642175674438477
RMSE :  4.652114577781742
r2 :  0.7410695204584946


loss :  19.890905554161282
mse :  19.89090347290039
RMSE :  4.459921807509031
r2 :  0.7620219728563189

loss :  21.02914579783557
mse :  21.029144287109375
RMSE :  4.5857543423941145
r2 :  0.7484038781108155
'''