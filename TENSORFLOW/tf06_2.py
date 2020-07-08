import tensorflow as tf

tf.set_random_seed(777)


x_train = tf.placeholder(dtype=tf.float32, shape=[None])
y_train = tf.placeholder(dtype=tf.float32, shape=[None])

# x_train = [1,2,3]
# y_train = [3,5,7]

# feed_dict = {x_train_hold:x_train, y_train_hold:y_train}


W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) 
# print(sess.run(W))

hypothesis = x_train * W + b       

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  

train = tf.train.GradientDescentOptimizer(learning_rate=0.12).minimize(cost) 

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())


    for step in range(601):    
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})  

        if step % 20 == 0:  
            print(step, cost_val, W_val, b_val)

    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[4]}))        
    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[5, 6]}))      
    print('예측: ', sess.run(hypothesis, feed_dict={x_train:[6, 7, 8]}))

# 500 lr = 0.01
# 예측:  [9.002827]
# 예측:  [11.004463 13.006101]
# 예측:  [13.006101 15.007738 17.009375]

# 500 lr = 0.02
# 예측:  [9.000845]
# 예측:  [11.001335 13.001823]
# 예측:  [13.001823 15.002314 17.002802]

# 500 lr = 0.03
# 예측:  [9.000252]
# 예측:  [11.000398 13.000544]
# 예측:  [13.000544  15.0006895 17.000835 ]

# 500 lr = 0.04
# 예측:  [9.000074]
# 예측:  [11.000117 13.000161]
# 예측:  [13.000161 15.000204 17.000248]

# 500 lr = 0.05
# 예측:  [9.000022]
# 예측:  [11.000035 13.000048]
# 예측:  [13.000048 15.00006  17.000074]

# 500 lr = 0.06
# 예측:  [9.000007]
# 예측:  [11.0000105 13.000014 ]
# 예측:  [13.000014 15.000018 17.000021]

# 500 lr = 0.07
# 예측:  [9.000003]
# 예측:  [11.000005 13.000006]
# 예측:  [13.000006 15.000008 17.00001 ]

# 500 lr = 0.08, 0.09
# 예측:  [9.000002]
# 예측:  [11.000003 13.000005]
# 예측:  [13.000005 15.000006 17.000008]

# 500 lr = 0.1
# 예측:  [9.000002]
# 예측:  [11.000003 13.000004]
# 예측:  [13.000004 15.000005 17.000006]

# 500 lr = 0.11
# 예측:  [9.000001]
# 예측:  [11.000002 13.000002]
# 예측:  [13.000002 15.000003 17.000004]

# 500 lr = 0.12
# 예측:  [9.000001]
# 예측:  [11.000002 13.000002]
# 예측:  [13.000002 15.000003 17.000004]