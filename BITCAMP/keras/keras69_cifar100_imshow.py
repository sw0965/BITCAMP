from keras.datasets import cifar100
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.dtype)

'''
print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[70])
plt.show()
'''