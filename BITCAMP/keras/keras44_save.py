import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM



#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4, 1)))     
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(333))
model.add(Dense(444))
model.add(Dense(555))
model.add(Dense(666))
model.add(Dense(777))

model.summary()

 #..은 현재폴더

# model.save(".//model//save_keras44.h5")  
# model.save(".//model//save_keras44.h5")
model.save(".\model\save_keras44.h5")
#3개 다 됨


print("저장 잘됬다.")