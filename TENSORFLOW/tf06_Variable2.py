# hypothesis
# H = Wx + b


import tensorflow as tf

tf.set_random_seed(777)


W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


x = tf.Variable([1,2,3], dtype=tf.float32)
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W*x + b)
print("hypothesis: ", aaa)
# print(aaa)
sess.close()

hypothesis = W*x + b

sess = tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print("hypothesis: ", bbb)
# print(bbb)
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)  
print("hypothesis: ", ccc)
# print(ccc)
sess.close()

