import numpy as np 
import matplotlib.pyplot as plt

from keras.datasets import mnist  #datasets  = 케라스에 있는 예제파일들

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)  #디멘션 하나
print(y_test.shape)


print(x_train[0].shape)
plt.imshow(x_train[59999], 'gray') 
# plt.imshow(x_train[0])
plt.show()