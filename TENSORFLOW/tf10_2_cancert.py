import tensorflow as tf
from sklearn.datasets import load_breast_cancer

tf.set_random_seed(777)

dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target.reshape(569, 1)

print(x_data.shape)     # (569, 30)
print(y_data.shape)     # (569, 1)
# print(y_data)

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.zeros([30, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) *tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5e-6)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        
        if step % 20 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print("\n\n Hypothesis :", "\n",h, "\n\n Predict :","\n",c, "\n\n accuracy :", a)