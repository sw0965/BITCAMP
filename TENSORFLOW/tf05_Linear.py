import tensorflow as tf

tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [3,5,7]

print(x_train.shape)

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수는 항상 초기화를 하고 실행 해야된다.
# print(sess.run(W))

hypothesis = x_train * W + b        # == y = wx + b (모델)

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # loss 값 계산 ['mse']

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)    # optimizer

with tf.Session() as sess:  # with문 안에 session이 다 실행됨
    sess.run(tf.global_variables_initializer())
    # 초기화를 하는 이유는 시작값을 잡아주어야 해서 초기화를 시킴


    for step in range(2001):    # 2000번 돌려라
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])   # _ = train, cost_val = loss , W_val = W, b_val = b (list형태로 된것들을 출력 하겠다)

        if step % 20 == 0:  # 20번마다 프린트 해줘라는 뜻
            print(step, cost_val, W_val, b_val)
