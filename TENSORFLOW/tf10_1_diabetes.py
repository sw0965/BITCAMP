import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

tf.set_random_seed(777)

dataset = load_diabetes()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(442,1)
print(type(x_data))     # (442, 10)
print(y_data.shape)     # (442,)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b    # wx + b    5, 3 * 3, 1 = 5, 1

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0099)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})

    if step % 200 == 0:
        print(step, "cost:", cost_val, "\n 실제값:",y_data, "예측값:", hy_val)