# for i in [1, 2, 3, 4, 5]:
#     print('only i :', i)
#     for j in [1, 2, 3, 4, 5]:
#         print('only j :', j)
#         print('i+j: ',i+j)
#     print(i)

# def my_print(message = "hi"):
#     print(message)

# my_print("hello")
# my_print()

# first_name = "Han"
# last_name = "Sangwoo"
# full_name1 = first_name + ""+last_name
# print(full_name1)

# interger_list = [1,2,3]
# heter = ["string", 0.1, True]
# l_o_l = [interger_list, heter, []]
# print(l_o_l)

# x, y, z = [1, 2, 'hi']
# print(x, y, z)


### 1. 데이터
import numpy as np

x_train = np.arange(1,1001,1)
y_train = np.array([1,0]*500)
# print(x_train)
# print(y_train.shape)

from keras.utils.np_utils import to_categorical
### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

y_train =to_categorical(y_train)
# print(y_train.shape)

model = Sequential()

model.add(Dense(32,activation="relu",input_shape=(1,)))
model.add(Dense(64,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


### 3. 실행, 훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)


### 4. 평가, 예측
loss = model.evaluate(x_train, y_train )
print('loss :', loss)

x_pred = np.array([11, 12, 13, 14])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
