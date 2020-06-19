import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )


print(x_train[0].shape)  #(28, 28)
# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# # plt.show()

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)


from keras.models import load_model

model = load_model('./model/model_test01.h5')
model.add(Dense(50 , name='dense1_1' ,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10 , name='dense1_2' ,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10 , name='dense1_3', activation='softmax'))



model.summary()


#4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=1)

loss, acc = loss_acc
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test)




y_pre  = np.argmax(y_predict,axis=-1)
y_test = np.argmax(y_test,axis=-1)

print(f"y_test[0:20]:{y_test[0:20]}")
print(f"y_pre[0:20]:{y_pre[0:20]}")

# print(y_test)
# print(y_predict)

''' load 파일
loss :  0.18315951786566817
acc :  0.9811999797821045
'''

''' 레이어 3개 추가
loss :  2.3136667798757555
acc :  0.09780000150203705
'''