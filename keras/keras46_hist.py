import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 101)) 
size = 5                           # time_steps = 4
print("a.shape :", a.shape)

# 실습: lstm 모델을 완성하시오. size 5


def split_x(seq, size):    #size = lstm의 timesteps (열)
    aaa = []
    for i in range(len(seq) - size + 1): #<-이게 행 길이에서 - size + 1 = 열 
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])   #가장중요
    print(type(aaa))
    return np.array(aaa)   #리턴값이 numpy라 밑에 type(dataset)이


dataset = split_x(a, size)
print("============================")
print(dataset)
print(dataset.shape)   #(6, 5) 
print(type(dataset))   #class 'numpy.ndarray' 

x = dataset[:, 0:4]    # : 은 all [모든행이 들어가겠다, (0,1,2,3)] 6, 5 짜리 똔똔해서 모든행을 가져오고 0부터 3(4-1) 까지를 가져오겠다.
y = dataset[:, 4]

print(x.shape)
print(y.shape)

x = np.reshape(x, (96, 4, 1))   #둘다 같은거임
# x2 = x.reshape(6, 4, 1)
# print(x1.shape)
# print(x2.shape)
# x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)


#2. 모델구성
from keras.models import load_model
model = load_model('./model/save_keras44.h5')
model.add(Dense(660, name="le1"))
model.add(Dense(500, name="le2"))
model.add(Dense(400, name="le3"))
model.add(Dense(300, name="le4"))
model.add(Dense(200, name="le5"))
model.add(Dense(100, name="le6"))
model.add(Dense(80, name="le7"))
model.add(Dense(30, name="le8"))
model.add(Dense(10, name="le9"))
model.add(Dense(1, name="le10"))





model.summary()


#3. 실행
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto') 

model.compile(optimizer='adam', loss='mse', metrics=['acc'])
hist = model.fit(x, y, epochs=1000, validation_split=0.2, batch_size=1, 
                       verbose=2, callbacks=[early_stopping])
#hist

print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])  # 선긋기
plt.title('loss & acc')        # 제목
plt.ylabel('loss, acc')        # y축 이름
plt.xlabel('epoch')            # x축 이름
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()
'''
loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)

print('loss : ', loss)
print('mse :', acc)
print('y_predict :', y_predict)
'''