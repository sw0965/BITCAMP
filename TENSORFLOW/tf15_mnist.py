# from keras.datasets import mnist
# import tensorflow as tf
# from collections import Counter
# # print(Counter(mnist))


# # tf.set_random_seed(777)

# # print(mnist.load_data())

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# a = Counter(y_train), Counter(y_test)
# print(a)

# # print(x_train.shape)    # (60000, 28, 28)
# # print(y_train.shape)    # (60000, )
# # print(x_test.shape)     # (10000, 28, 28)
# # print(y_test.shape)     # (10000, )
# def init_weights(shape):
#     return tf.Variable(tf.random_normal(shape))

# def model(x, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    
#     l1a = tf.nn.relu(tf.nn.conv2d(x, w1,                       # l1a shape=(?, 28, 28, 32)
#                         strides=[1, 1, 1, 1], padding='SAME'))
#     l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
#                         strides=[1, 2, 2, 1], padding='SAME')
#     l1 = tf.nn.dropout(l1, p_keep_conv)

#     l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
#                         strides=[1, 1, 1, 1], padding='SAME'))
#     l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
#                         strides=[1, 2, 2, 1], padding='SAME')
#     l2 = tf.nn.dropout(l2, p_keep_conv)

#     l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
#                         strides=[1, 1, 1, 1], padding='SAME'))
#     l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
#                         strides=[1, 2, 2, 1], padding='SAME')
#     l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
#     l3 = tf.nn.dropout(l3, p_keep_conv)

#     l4 = tf.nn.relu(tf.matmul(l3, w4))
#     l4 = tf.nn.dropout(l4, p_keep_hidden)

#     pyx = tf.matmul(l4, w_o)
#     return pyx



# # one_hot 
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# y_train = tf.one_hot(y_train, depth=10).eval(session=sess)
# y_test = tf.one_hot(y_test, depth=10).eval(session=sess)

# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# print(x_train.shape)    # (60000, 784)
# print(y_train.shape)    # (60000, 10)
# print(x_test.shape)     # (10000, 784)
# print(y_test.shape)     # (10000, 10)


# x = tf.placeholder(dtype=float, shape=[None, 28, 28, 1])
# y = tf.placeholder(dtype=float, shape=[None, 10])



# w1 = init_weights([28, 28, 1, 32])       # 3x3x1 conv, 32 outputs
# w2 = init_weights([28, 28, 32, 128])     # 3x3x32 conv, 64 outputs
# w3 = init_weights([28, 28, 128, 64])    # 3x3x32 conv, 128 outputs
# w4 = init_weights([28, 28, 64, 32]) # FC 128 * 4 * 4 inputs, 625 outputs
# w_o = init_weights([32, 10]) 

# p_keep_conv = tf.placeholder("float")
# p_keep_hidden = tf.placeholder("float")

# py_x = model(x, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, y))
# opt = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)

# pred = tf.argmax(py_x, 1)

# with tf.Session() as sess:
#     sess.run(tf.random_normal_initializer())

#     for step in range(2001):
#         sess.run(opt, feed_dict={x:x_train, y:y_train})


import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

x_train = x_train.reshape(-1, 28, 28, 1)  # 28x28x1 input img
x_test = x_test.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(x_train), batch_size),
                             range(batch_size, len(x_train)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: x_train[start:end], Y: y_train[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(x_test)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(y_test[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: x_test[test_indices],
                                                         Y: y_test[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

