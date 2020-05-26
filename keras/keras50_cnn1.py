from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) #(9, 9, 10)
#(2,2)=2,2로 짜르겠다. 인풋 끝에 1은 명암 1은 흑백 3은 칼라
#모든 이미지는 (10,10,1)
model.add(Conv2D(5, (2,2), padding='same'))                            # (7, 7, 7)
model.add(Conv2D(5, (2,2), padding='same'))            # (7, 7, 5)
model.add(Conv2D(5, (2,2), padding='same'))                            # (6, 6, 5)
# model.add(Conv2D(5, (2,2), strides=2))               # (3, 3, 5)
# model.add(Conv2D(5, (2,2), strides=2, padding='same')) # (3, 3, 5)
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())  # 쫙 피는거 (3,3) 3장있으면 가로로 이어서 9개
# model.add(Dense(1))


model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 9, 9, 10)          50   

_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 7)           637  

_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 5)           145  

_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 5)           105  

_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 5)           0    

_________________________________________________________________
flatten_1 (Flatten)          (None, 45)45는 위에거 다 곱한거0    

_________________________________________________________________
dense_1 (Dense)              (None, 1)                 46  
'''