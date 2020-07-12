from keras.datasets import mnist
from collections import Counter
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

 # one_hot 
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
# x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255
x_train = x_train.reshape(-1, 28*28).astype('float32')/255
x_test = x_test.reshape(-1, 28*28).astype('float32')/255

print(x_train.shape)    # (60000, 784)
print(y_train.shape)    # (60000, 10)
print(x_test.shape)     # (10000, 784)
print(y_test.shape)     # (10000, 10)


LEARNING_RATE = 1e-3
TRAIN_EPOCH = 15
BATCH_SIZE = 100
TOTAL_BATCH = int(len(x_train) / BATCH_SIZE)    # 60000 / 100

x = tf.placeholder('float32', shape=[None, 784])
y = tf.placeholder('float32', shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)  # dropout

w1 = tf.get_variable("w1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(tf.matmul(x,w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


w2 = tf.get_variable("w2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1,w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

w3 = tf.get_variable("w3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2,w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable("w4", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3,w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

w5 = tf.get_variable("w5", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
h = tf.nn.selu(tf.matmul(L4,w5) + b5)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h),axis=1)) # loss ... 계산 방법 ...

opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(TRAIN_EPOCH):              # 15
    ave_cost = 0

    for i in range(TOTAL_BATCH):                # 600
        batch_xs, batch_ys = 