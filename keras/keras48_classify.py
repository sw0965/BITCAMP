import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])   # 이진분류 

# print(x.shape)
# print(x)
# print(y.shape)
# print(y)



#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#activation은 각 노드 값에 곱해주는 것 그래서 마지막에 sigmoid로 빼준다 0아니면 1로 나오는거.

    

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc']) 
#0인데 1 나올수도 있으니 mertrics에서 accuracy를 쓴다.
#로스값은 이진분류 하게되면 binary 저거 하나밖에 없음.
model.fit(x,y,epochs=184,batch_size=16,verbose=2)#, callbacks=[early_stopping])

#4 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)
print('acc : ', acc)

x_pred = np.array([1, 2, 3])
y_pred = model.predict(x_pred)
print(y_pred)

for i in range(len(y_pred)):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

print(y_pred)

'''
loss :  0.5017561411626957
acc :  0.6000000238418579
[[9.9996579e-01]
 [6.1704713e-04]
 [8.9712453e-01]]
[[1.]
 [0.]
 [1.]]
 '''