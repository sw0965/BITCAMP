import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 과제 1) 전부다 sequential로 80프로이상
# 64, 65 66 제일 하단에 주석으로 acc와 loss 결과 명시

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[50000])
plt.show()
