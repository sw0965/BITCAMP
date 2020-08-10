# # a2~5는 auto encorder가 아님 1, 6이 autoencoder

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv2D(filters=32 , kernel_size=(3,3), padding='same', input_shape=(28, 28, 1), activation='relu'))
#     # model.add(Flatten())
#     model.add(Conv2D(filters= hidden_layer_size, kernel_size=(3, 3), padding= 'same', activation='relu'))
#     # model.add(Dense(units=64, activation='relu'))
#     # model.add(Dense(units=128, activation='relu'))
#     # model.add(Dense(units=784, activation='sigmoid'))
#     return model

# from tensorflow.keras.datasets import mnist

# train_set, test_set = mnist.load_data()
# x_train, y_train = train_set
# x_test, y_test = test_set

# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# x_train = x_train/255.
# x_test = x_test/255.

# print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# model = autoencoder(hidden_layer_size=154)

# # model.compile(optimizer='adam', loss='mse', metrics=['acc'])                   # 32 loss = 0.0102, acc = 0.0131 나옴 ..
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])     # 32 loss = 0.0936, acc = 0.8142 나왔는데 acc 보고 판단말고 loss 를 보고 판단하여라 - 쌤
# model.fit(x_train, x_train, epochs=10)
# output = model.predict(x_test)

# from matplotlib import pyplot as plt
# import random
# fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# # 이미지 다섯 개를 무작위로 고른다. 
# random_images = random.sample(range(output.shape[0]), 5)

# # 원본(입력) 이미지를 맨 위에 그린다.
# for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
#     ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
#     if i ==0 : 
#         ax.set_ylabel("INPUT", size=40)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# # 오토 인코더가 출력한 이미지를 아래에 그린다.
# for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
#     ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
#     if i ==0 : 
#         ax.set_ylabel("OUTPUT", size=40)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.show()

# 154 loss: 0.0658 - acc: 0.8155
'''
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
loss: 0.0740 - acc: 0.8153

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=356, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
loss: 0.0955 - acc: 0.8132
'''


##############################

# a08_ae4_cnn 복붙
# cnn으로 오토인코더 구성하시오

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.datasets import mnist

def autoencoder(middle, hidden):
    model = Sequential()
    # model.add(Conv2D(filters=hidden, kernel_size=(3,3), padding = 'same', input_shape=(28,28,1), activation='relu'))
    # model.add(Conv2D(filters=middle, kernel_size=(2,2), padding = 'same', activation='relu'))
    # model.add(Conv2D(filters=hidden, kernel_size=(2,2), padding = 'same', activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(units=1, activation='sigmoid'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='relu', input_shape=(28, 28, 1), padding='valid'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7*7*64, activation='relu'))
    model.add(Reshape(target_shape=(7,7,64)))
    model.add(Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=2, strides=(2,2), padding='same', activation='sigmoid'))
    model.summary()
    return model

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train/255
x_test = x_test/255


model = autoencoder(middle=154, hidden=308)

model.compile(optimizer='adam', loss='mse',metrics=['acc']) #loss = 0.01 () 0.002
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc']) #loss = 0.09 () 0.06


model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

#이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_xlabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("output", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

##################################################################
# a08_ae4_cnn 복붙
# cnn으로 오토인코더 구성하시오
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size*3, kernel_size=(3,3), padding='valid', input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters=hidden_layer_size*2, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(filters=hidden_layer_size*1, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(3,3)))
    model.add(Conv2D(filters=hidden_layer_size*2, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(UpSampling2D(size=(3,3)))
    model.add(Conv2D(filters=hidden_layer_size*3, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid'))
    model.summary()
    return model

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train/255
x_test = x_test/255

model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='mse',metrics=['acc']) #loss = 0.01 () 0.002
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc']) #loss = 0.09 () 0.06


model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

#이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_xlabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("output", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
