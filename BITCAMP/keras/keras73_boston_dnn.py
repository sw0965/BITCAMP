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
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state = 66, shuffle=True, train_size=0.7)

print('x_train.shape : ',x_train.shape)  # (354, 10)
print('x_test.shape : ',x_test.shape)    # (152, 10)
print('y_train.shape : ',y_train.shape)  # (354, )
print('y_test.shape : ',y_test.shape)    # (152, )


# 모델 구성
model = Sequential()
model.add(Dense(10, input_shape=(10,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
# model.save('./model/sample/boston/model_boston.h5')
# 훈련

# early_stopping = EarlyStopping(monitor='acc', patience=2, mode='auto')
# model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) 
# model.fit(x_train,y_train,epochs=1000,batch_size=1,verbose=1 ,validation_split=0.4)# ,callbacks=[early_stopping])

es            = EarlyStopping(monitor='mse', patience=2, mode='auto')
modelpath     = './model/sample/boston/boston-{epoch:02d}-{val_mse:.4f}.hdf5'
cp            = ModelCheckpoint(filepath=modelpath, monitor='val_mse', save_best_only=True, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist          = model.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=16, verbose=1, callbacks=[es,cp])
# D:\Study\study\graph>cd tensorboard --logdir=.(텐서보드 확인 cmd에서)
# model.save_weights('./model/sample/boston/weight_boston.h5')




#. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)


y_pred = model.predict(x_test)
print(y_pred)


#RMSE 구하기 #낮을수록 좋다
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)


# 튜닝필요