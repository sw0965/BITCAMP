import numpy as np 
from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])
# print('y_train :', y_train[0]) # 5

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (10000, 28, 28)  
print(y_train.shape)  # (60000, )디멘션 하나
print(y_test.shape)   # (10000, )


print(x_train[0].shape)  #(28, 28)
# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# # plt.show()

#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)


np.save('./data/mnist_train_x.npy', arr=x_train)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_x.npy',  arr=x_test)
np.save('./data/mnist_test_y.npy',  arr=y_test)
