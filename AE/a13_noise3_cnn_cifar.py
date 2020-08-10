# cifar10 으로 autoencoder 구성할 것.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Reshape, Conv2DTranspose, Flatten
from tensorflow.keras.datasets import cifar10

train_set, test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 1) (10000, 1)

def autoencoder(hidden):
    model = Sequential()
    model.add(Conv2D(filters= hidden, kernel_size=2, strides=(2,2), padding='valid', input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(filters=128 , kernel_size=2, strides=(2,2), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256*2, activation='relu'))
    model.add(Dense(units=8*8*256*2, activation='relu'))
    model.add(Reshape(target_shape=(8, 8, 256*2)))
    model.add(Conv2DTranspose(filters=256*2, kernel_size=2, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2DTranspose(filters=3, kernel_size=2, strides=(2, 2), padding='valid', activation='sigmoid'))
    return model
# 스케일링
x_train = x_train/255.
x_test = x_test/255.
print(x_train)

model = autoencoder(hidden=64)

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32, 32, 3), cmap='gray')

    if i ==0 : 
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])



# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(32,32,3), cmap='gray')

    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()