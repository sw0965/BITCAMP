import numpy as np
from keras.utils import np_utils
import pandas as pd
from keras.datasets import cifar10
import tensorflow as tf
from collections import Counter
tf.set_random_seed(777)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

print(x_train.__class__)    # numpy

# print(y_train)
# print(y_test)
# y = y_train, y_test

# Cateogotical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
KEEP_PROB = tf.placeholder(tf.float32)

# Model
# input layer
W1 = tf.get_variable("w1", shape=[3,3,3,32])
print("w1: ", W1)   # shape=(3, 3, 3, 32)
L1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')
print(L1)   # shape=(?, 32, 32, 32)
L1 = tf.nn.relu(L1)
print(L1)   # shape=(?, 32, 32, 32)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')
print(L1)   # shape=(?, 16, 16, 32)
L1 = tf.nn.dropout(L1, keep_prob=KEEP_PROB)

# hidden layer
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1], strides=[1, 2,2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=KEEP_PROB)

# hidden layer
W3 = tf.get_variable("W3", shape=[3, 3, 64, 43])
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)   # shape=(?, 4, 4, 43)

L3_FLATTEN = tf.reshape(L3, [-1, 4*4*43])
print(L3_FLATTEN.shape)
W4 = tf.get_variable("W4", shape=[688, 10], initializer = tf.contrib.layers.xavier_initializer())
B4 = tf.Variable(tf.random_normal([10]))

HYPOTHESIS = tf.nn.softmax(tf.matmul(L3_FLATTEN, W4) + B4)

LEARNING_RATE = 0.1

# SET VALUE
COST = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(HYPOTHESIS), axis = 1))
OPTIMIZER = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(COST)
PREDICTION = tf.equal(tf.math.argmax(HYPOTHESIS, 1), tf.arg_max(y, 1))
ACCURACY = tf.reduce_mean(tf.cast(PREDICTION, tf.float32))


TRAINING_EPOCHS = 15
BATCH_SIZE = 100
TOTAL_BATCH = int(len(x_train) / BATCH_SIZE)    # 500

print(TOTAL_BATCH)

def next_batch(num, data, labels):
       '''
       Return a total of `num` random samples and labels. 
       '''
       idx = np.arange(0 , len(data))
       np.random.shuffle(idx)
       idx = idx[:num]
       data_shuffle = [data[i] for i in idx]
       labels_shuffle = [labels[i] for i in idx]

       return np.asarray(data_shuffle), np.asarray(labels_shuffle)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0

        for i in range(TOTAL_BATCH):
            batch_xs, batch_ys = next_batch(100, x_train, y_train)
            feed_dict = {x:batch_xs, y:batch_ys, KEEP_PROB:1}
            c, _, acc_val = sess.run([COST, OPTIMIZER, ACCURACY], feed_dict = feed_dict)
            avg_cost += c / TOTAL_BATCH
        print("Epoch : ", "%4d" % (epoch+1) , "cost = " , "{:.9f}".format(avg_cost))

    FEED_DICT_TEST = {x:x_test, y:y_test, KEEP_PROB : 0.7}
    ACC = sess.run([ACCURACY], feed_dict= FEED_DICT_TEST)

    print('훈련 끝')
    CORRECT_PREDICTION = tf.equal(tf.argmax(HYPOTHESIS,1), tf.argmax(y,1))
    ACCURACY = tf.reduce_mean(tf.cast(CORRECT_PREDICTION, tf.float32))
    _, acc = sess.run([HYPOTHESIS,ACCURACY], feed_dict={x: x_test, y: y_test, KEEP_PROB : 0.6})
    
    print('Acc : ',acc ) ## acc 출력