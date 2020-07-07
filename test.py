'''# for i in [1, 2, 3, 4, 5]:
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
'''

# a = [['alice', [1, 2, 3]], ['bob', 20], ['tony', 15], ['suzy', 30]]
# b = dict(a)
# print(b)
# print(b['alice'][1])

# a = 10
# ls = []
# ls.append(a)
# print(ls)
# while a:
#     b = a*10
#     ls.append(b)
#     if b == 10000:
#         break
#     print(ls)



from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img


img_dog = load_img('./DATA/dog_cat/dog.jpg', target_size=(224, 224))
img_cat = load_img('./DATA/dog_cat/cat.jpg', target_size=(224, 224))
img_suit = load_img('./DATA/dog_cat/suit.jpg', target_size=(224, 224))
img_onion = load_img('./DATA/dog_cat/onion.jpg', target_size=(224, 224))

plt.imshow(img_suit)
plt.imshow(img_onion)
plt.imshow(img_dog)
plt.imshow(img_cat)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_onion = img_to_array(img_onion)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# RGB -> BGR
from keras.applications.vgg19 import preprocess_input

# 데이터 전처리
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_onion = preprocess_input(arr_onion)

print(arr_dog)

# 이미지를 하나로 합친다.
import numpy as np 
arr_input = np.stack([arr_dog, arr_cat, arr_onion, arr_suit])
print(arr_input.shape)  # (4, 224, 224, 3)

# 모델 구성
model = VGG19()
probs = model.predict(arr_input)

print(probs)

print('probs.shape: ', probs.shape) # probs.shape:  (4, 1000)

# 이미지 결과
from keras.applications.vgg19 import decode_predictions

results = decode_predictions(probs)

print('-------------------')
print(results[0])
print('-------------------')
print(results[1])
print('-------------------')
print(results[2])
print('-------------------')
print(results[3])
# decode_predictions 하면 이렇게 됌.
