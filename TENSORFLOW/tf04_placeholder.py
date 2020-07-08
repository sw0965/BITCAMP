import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)

adder_node = a + b
adder_node1 = a + c


print(sess.run([adder_node,adder_node1], feed_dict={a:3, b:4.5, c:2}))     # sses run 하는 시점에 a + b 를 넣는데 feed_dict= input 개념
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

add_and_triple = adder_node *3

# print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))