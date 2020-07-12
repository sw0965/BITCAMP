from keras.datasets import mnist
import tensorflow as tf
from collections import Counter
# print(Counter(mnist))


# tf.set_random_seed(777)

# print(mnist.load_data())

(x_train, y_train), (x_test, y_test) = mnist.load_data()
a = Counter(y_train), Counter(y_test)
print(a)

# print(x_train.shape)    # (60000, 28, 28)
# print(y_train.shape)    # (60000, )
# print(x_test.shape)     # (10000, 28, 28)
# print(y_test.shape)     # (10000, )


# one_hot 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_train = tf.one_hot(y_train, depth=10).eval(session=sess)
y_test = tf.one_hot(y_test, depth=10).eval(session=sess)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape)    # (60000, 784)
print(y_train.shape)    # (60000, 10)
print(x_test.shape)     # (10000, 784)
print(y_test.shape)     # (10000, 10)



x = tf.placeholder(dtype=float, shape=[None, 28*28])
y = tf.placeholder(dtype=float, shape=[None, 1])

w = tf.Variable(tf.random_normal([784, 30]), name = 'w')
b = tf.Variable(tf.random_normal([30]), name = 'b')
layer = tf.nn.softmax(tf.matmul(x,w) + b)
