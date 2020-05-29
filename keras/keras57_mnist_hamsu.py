import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, Input
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

#0 부터 9까지 분류 onehotencording
#데이터 전처리 1. 원핫인코딩
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)
'''
#데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(x_test)
# print(x_train)
# print(y_test)
# print(y_train)
# 모델구성

input1 = Input(shape=(28, 28, 1))
ic1 = Conv2D(30, (2,2), activation='relu', padding='same')(input1) 
ic2 = Conv2D(30, (2,2),activation='relu', padding='same')(ic1)   
ic3 = Conv2D(30, (2,2),activation='relu', padding='same')(ic2)
oc1 = Conv2D(30, (2,2),activation='relu', padding='same')(ic3)
drop = Dropout(0.3) (oc1)
maxp = MaxPooling2D(pool_size=2)(drop)
flat1 = Flatten()(maxp) 
den1 = Dense(784, activation='relu')(flat1)
den2 = Dense(10, activation='softmax')(den1)

model = Model(inputs = input1, outputs = den2)
# model.add(Flatten())    
# model.add(MaxPooling2D(pool_size=2))
# model.add(Activation('relu'))

model.summary()

#3. 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train,y_train,epochs=17,batch_size=128,validation_split=0.2,verbose=1)# callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss : ', loss)
print('acc : ', acc)
# print(x_test)
# print(y_test)


# y_predict = model.predict(x_test)




# y_pred = np.argmax(y_pred,axis=1) + 1
# print(y_pred)

# print(y_test)
# print(y_predict)


# onehotencording
# 분류때 남자는 1 여자는 2일때 남자 + 남자 = 여자가 아니므로 onehotencording을 사용
# 분류하는 갯수 만큼 shape가 늘어남 
'''
'''
loss :  0.0074524702965347205
acc :  0.9978694319725037
'''
