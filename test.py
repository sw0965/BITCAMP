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



# from keras.applications.vgg19 import VGG19
# import matplotlib.pyplot as plt

# from keras.preprocessing.image import load_img


# img_dog = load_img('./DATA/dog_cat/dog.jpg', target_size=(224, 224))
# img_cat = load_img('./DATA/dog_cat/cat.jpg', target_size=(224, 224))
# img_suit = load_img('./DATA/dog_cat/suit.jpg', target_size=(224, 224))
# img_onion = load_img('./DATA/dog_cat/onion.jpg', target_size=(224, 224))

# plt.imshow(img_suit)
# plt.imshow(img_onion)
# plt.imshow(img_dog)
# plt.imshow(img_cat)
# # plt.show()

# from keras.preprocessing.image import img_to_array

# arr_dog = img_to_array(img_dog)
# arr_cat = img_to_array(img_cat)
# arr_onion = img_to_array(img_onion)
# arr_suit = img_to_array(img_suit)

# print(arr_dog)
# print(type(arr_dog))
# print(arr_dog.shape)

# # RGB -> BGR
# from keras.applications.vgg19 import preprocess_input

# # 데이터 전처리
# arr_dog = preprocess_input(arr_dog)
# arr_cat = preprocess_input(arr_cat)
# arr_suit = preprocess_input(arr_suit)
# arr_onion = preprocess_input(arr_onion)

# print(arr_dog)

# # 이미지를 하나로 합친다.
# import numpy as np 
# arr_input = np.stack([arr_dog, arr_cat, arr_onion, arr_suit])
# print(arr_input.shape)  # (4, 224, 224, 3)

# # 모델 구성
# model = VGG19()
# probs = model.predict(arr_input)

# print(probs)

# print('probs.shape: ', probs.shape) # probs.shape:  (4, 1000)

# # 이미지 결과
# from keras.applications.vgg19 import decode_predictions

# results = decode_predictions(probs)

# print('-------------------')
# print(results[0])
# print('-------------------')
# print(results[1])
# print('-------------------')
# print(results[2])
# print('-------------------')
# print(results[3])
# # decode_predictions 하면 이렇게 됌.

# import tensorflow as tf

# tf.set_random_seed(777)

# x_train = tf.placeholder(dtype=tf.float32, shape=[None])
# y_train = tf.placeholder(dtype=tf.float32, shape=[None])

# # x_train = [1,2,3]
# # y_train = [3,5,7]

# # feed_dict = {x_train_hold:x_train, y_train_hold:y_train}


# W = tf.Variable(tf.random_normal([1]), name='Weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# # sess = tf.Session()
# # sess.run(tf.global_variables_initializer()) 
# # print(sess.run(W))

# hypothesis = x_train * W + b       

# cost = tf.reduce_mean(tf.square(hypothesis - y_train))  

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) 

# with tf.Session() as sess:  
# # with tf.compat.v1.Session as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(hypothesis, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
#     sess.run(cost, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
#     sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})


#     for step in range(2001):    
#         _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  

#         if step % 20 == 0:  
#             print(step, cost_val, W_val, b_val)

# import tensorflow as tf

# tf.set_random_seed(777)


# x_train = tf.placeholder(dtype=tf.float32, shape=[None])
# y_train = tf.placeholder(dtype=tf.float32, shape=[None])

# # x_train = [1,2,3]
# # y_train = [3,5,7]

# # feed_dict = {x_train_hold:x_train, y_train_hold:y_train}


# W = tf.Variable(tf.random_normal([1]), name='Weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# # sess = tf.Session()
# # sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))

# hypothesis = x_train * W + b       

# cost = tf.reduce_mean(tf.square(hypothesis - y_train))  

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) 

# with tf.Session() as sess:  
# # with tf.compat.v1.Session as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(hypothesis, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
#     sess.run(cost, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
#     sess.run(train, feed_dict={x_train:[1,2,3], y_train:[3,5,7]})


#     for step in range(2001):    
#         _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  

#         if step % 20 == 0:  
#             print(step, cost_val, W_val, b_val)


# import tensorflow as tf
# import numpy as np
# from sklearn.model_selection import train_test_split as tts
# from keras.datasets import mnist
# from keras.utils.np_utils import to_categorical

# # 데이터 입력
# (x_train,y_train),(x_test,y_test)=mnist.load_data()

# print(x_train.shape)#(60000, 28, 28)
# print(y_train.shape)#(60000,)

# #전처리1) - minmax
# x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2], 1)
# x_test = x_test.reshape(-1,x_test.shape[1],x_test.shape[2], 1)

# #전처리2) to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# learning_rate=0.001
# traing_epochs=35
# batch_size=100
# total_batch = x_train.shape[0]//batch_size #600

# print()

# x = tf.placeholder(tf.float32, shape=[None,28, 28, 1])
# y = tf.placeholder(tf.float32, shape=[None,10])
# keep_prob = tf.placeholder(tf.float32)

# # w = tf.Variable(tf.zeros([28*28,10]),name="weight")
# w = tf.get_variable("weight1",[3, 3, 1, 32])
# b = tf.Variable(tf.random_normal([32]),name="bias1")
# layer = tf.nn.relu(tf.nn.conv2d(x,w, strides=[1,1,1,1], padding='SAME'))
# layer = tf.nn.max_pool2d(layer,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# layer = tf.nn.dropout(layer,rate=1 - keep_prob)


# # # tf.contrib.layers.
# w = tf.get_variable("weight2",[3,3,32,64])
# b = tf.Variable(tf.random_normal([64]),name="bias2")
# layer = tf.nn.relu(tf.nn.conv2d(layer,w,strides=[1,1,1,1], padding='SAME'))
# layer = tf.nn.max_pool2d(layer,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# layer = tf.nn.dropout(layer,rate=1 - keep_prob)

# w = tf.get_variable("weight3",[3,3,64,30])
# b = tf.Variable(tf.zeros([30]),name="bias3")
# layer = tf.nn.relu(tf.nn.conv2d(layer,w,[1,1,1,1], padding='SAME'))
# layer = tf.nn.max_pool2d(layer,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# layer = tf.nn.dropout(layer,rate=1 - keep_prob)
# print(layer.shape)  # (?, 4, 4, 30)

# flatten = tf.reshape(layer, [-1, 4*4*30])

# w = tf.get_variable("weight4",[4*4*30,10],initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.zeros([10]),name="bias4")

# layer = tf.matmul(flatten, w) + b


# # hypothesis = tf.nn.softmax(layer)

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(layer, y))

# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

# correct_prediction = tf.equal(tf.argmax(layer,1),tf.argmax(y,1))
# # #정확도
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     for epoch in range(traing_epochs):#15
#         avg_cost=0
#         start=0
#         for i in range(total_batch):
#             batch_xs,batch_ys = x_train[start:start+batch_size],y_train[start:start+batch_size]
#             start += batch_size
#             feed_dict = {x:batch_xs,y:batch_ys,keep_prob:0.7}
#             _,loss_val=sess.run([optimizer,loss],feed_dict=feed_dict)
#             avg_cost+=loss_val/total_batch

#         print(f"epoch:{epoch+1},loss_val:{avg_cost}")
#             # for i in range(total_batch):#600


#     print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_prob:0.7}))


# # # epoch:33,loss_val:0.08123232854297385
# # # epoch:34,loss_val:0.08043446005632474
# # # epoch:35,loss_val:0.0821741102013039
# # # Accuracy: 0.964

# import sys
# a = sys.stdin.read()

# import tensorflow as tf

# strategy = tf.distribute.MirroredStrategy()

# print('장치의 수 : {}'.format(strategy.num_replicas_in_sync))
import sys

N = int(sys.stdin.readline())
namelist = list()
for i in range(N):
    name = input()
    namelist.append(list(name))

transpose = [[namelist[j][i] for j in range(N)] for i in range(len(namelist[0]))]

sub = []
for word in transpose:
    print(word) 

    