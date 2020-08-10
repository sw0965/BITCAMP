# CNN으로 a11번 파일 작성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Reshape, Conv2DTranspose, Flatten
import numpy as np

def autoencoder(hidden1, hidden2):
    model = Sequential()
    model.add(Conv2D(filters=hidden1, kernel_size=2, strides=(2, 2), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(filters=hidden2, kernel_size=2, strides=(2, 2), padding= 'valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=7*7*80, activation='relu'))
    model.add(Reshape(target_shape=(7, 7, 80)))
    model.add(Conv2DTranspose(filters=hidden2, kernel_size=2, strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=2, strides=(2, 2), padding='valid', activation='sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

x_train = x_train/255.
x_test = x_test/255.

x_train_noised = x_train + np.random.normal(0, 0.2, size= x_train.shape) # randomnormal 정규분포에 의한 랜덤값. (0 평균 0.5 표준편차) 노이즈를 만들어 주기 위해 코드 작성
x_test_noised = x_test + np.random.normal(0, 0.2, size= x_test.shape)    # 위와 같이 test도 noised 작성
'''
점을 뿌려준거에 문제점이 있음. 
범위가 255로 나누어서 0 ~ 255 사이에 있음.
평균이 0 표준편차가 0.5면 음수가 들어갈수도 있는건 상관없지만 양수가 들어가면 0부터 정규분포니까 큰수로 들어갈 수 있음
0 ~ 1 사이에 우리가 지금 해놓은 스케일링 범위를 넘어간다 -> 의도하지 않은 대로 흘러가니 노이즈도 스케일링 해야함
'''
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
'''
NP.CLIP 설명

np.clip(배열, 최소값 기준, 최대값 기준) 을 사용하면 최소값과 최대값 조건으로 값을 기준으로 해서,
이 범위 기준을 벗어나는 값에 대해서는 일괄적으로 최소값, 최대값으로 대치해줄 때 매우 편리합니다. 
최소값 부분을 0으로 해주었으므로 0보다 작은 값은 모두 0으로 대치되었습니다. 
이때 원래의 배열 a는 그대로 있습니다. 

예시 
a = np.arange( -5 ,5 )
print(a) = ([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
np.clip(a, 0, 4) 
= ([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])

np.clip(배열, 최소값 기준, 최대값 기준, out 배열)을 사용해서 
out = a 를 추가로 설정해주면 반환되는 값을 배열 a에 저장할 수 있습니다. 
배열 a의 0보다 작았던 부분이 모두 0으로 대치되어 a가 변경되었음을 확인할 수 있습니다. 

예시
np.clop(a, 0, 4, out=a)
= ([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
print(a) = ([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])

'''

# print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

model = autoencoder(hidden1=64, hidden2=128)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])                   # 32 loss = 0.0102, acc = 0.0131 나옴 ..
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])     # 32 loss = 0.0936, acc = 0.8142 나왔는데 acc 보고 판단말고 loss 를 보고 판단하여라 - 쌤

model.fit(x_train_noised, x_train, epochs=10)   # 잡음이 있는걸 넣고 target을 잡음이 없는걸로 넣어서 훈련을 시킨다.

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
     (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))

# 이미지 다섯 개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    '''
    기본 이미지 (원래 test 이미지)
    '''
    if i ==0 : 
        ax.set_ylabel("ORIGNAL", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    '''
    노이즈 이미지 
    '''
    if i ==0 : 
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    '''
    x_test_noised를 넣고 predict한 이미지 
    '''
    if i ==0 : 
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()



