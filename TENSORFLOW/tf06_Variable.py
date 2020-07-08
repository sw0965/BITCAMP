import tensorflow as tf

tf.set_random_seed(777)


W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


W = tf.Variable([0.3], tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa= sess.run(W)
print(aaa)
sess.close()

sess = tf.InteractiveSession()  # sess.run(W) 대신 W.eval()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)  # 그냥 session 도 eval 이 먹힘 하지만 session을 언급 이렇게 언급 해줘야됌
print(ccc)
sess.close()

