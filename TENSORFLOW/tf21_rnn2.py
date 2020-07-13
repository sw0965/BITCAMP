import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape)    # (10, )


SIZE = 6

# SPLIT DATA
def split_x(seq, size):    #size = lstm의 timesteps (열)
    aaa = []
    for i in range(len(seq) - SIZE + 1): #<-이게 행 길이에서 - size + 1 = 열 
        subset = seq[i : (i + SIZE)]
        aaa.append([item for item in subset])   #가장중요
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, SIZE)
print(dataset)


# MAKE x_data, y_data
x_data = dataset[:,:5].reshape(1, 5, 5)
print(x_data.shape)     # (5, 5)
y_data = dataset[:,5:]
print(y_data.shape)     # (5, 1)

# y_data = np.argmax(y_data, axis=1)


# MAKE RNN MODEL
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 5, 5])
Y = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])

cell = tf.keras.layers.LSTMCell(5)
hypothesis, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
print(hypothesis)   # shape(none, 5, 1)

# COMPILE
weight = tf.ones([5, 1])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=hypothesis, 
    #targets=Y, 
    weights=weight)

cost = tf.compat.v1.reduce_mean(sequence_loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.compat.v1.argmax(hypothesis, axis=2)

# TRAINING
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, "loss : ", loss, "prediction : ", result, "true Y : ", y_data)

        # result_str = [dataset[c] for c in np.squeeze(result)]
        # print("\nPREDICTION STR : ", ''.join(result_str))