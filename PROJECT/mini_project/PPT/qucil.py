import numpy as np
from keras.models import Input, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model


#데이터 로드
x         = np.load('./data/mini_project/x_data.npy')
y         = np.load('./data/mini_project/y_data.npy')
x_predict = np.load('./data/mini_project/x_predict.npy')
print(x.shape)         #(160, 100, 100, 3)
print(y.shape)         #(160, 4)
print(x_predict.shape) #(20, 100, 100, 3)


# train test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
print(x_train.shape) #(128, 100, 100, 3)
print(x_test.shape)  #(32, 100, 100, 3)
print(y_train.shape) #(128 , 4)
print(y_test.shape)  #(32 , 4)


# 스케일러 위해 리쉐이프
x_train = x_train.reshape(128, 30000)
x_test = x_test.reshape(32, 30000)
x_predict = x_predict.reshape(20, 30000)

x_train = x_train/255
x_test = x_test/255
x_predict = x_predict/255


# 스케일러
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)


# 모델링을 위해 다시 리쉐이프
x_train = x_train.reshape(128, 100, 100, 3)
x_test = x_test.reshape(32, 100, 100, 3)
x_predict = x_predict.reshape(20, 100, 100, 3)


# 모델 불러오기
model = load_model('./mini_project/model_save/cnn_model_1.h5')


# 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


# y_predict 대입
y_predict = model.predict(x_test)
print(y_predict)
'''
y_predict = np.argmax(y_predict,axis=-1)
print(' 예측한 과일 : ',y_predict)  
print('')


for i in y_predict:
    if i == 0:
        print('사과 입니다.') 
        print('')
    elif i == 1:
        print('바나나 입니다.') 
        print('')
    elif i == 2:
        print('포도 입니다.')
        print('')
    elif i == 3:
        print('파인애플 입니다.')
        print('')


names  = ['사과', '바나나', '포도', '파인애플']
prices = [1200, 1800, 2900, 3800]
all_prices = 0
cnt    = [0,0,0,0]


for i in y_predict:
    if i == 0:
        cnt[0] += 1
    elif i == 1:
        cnt[1] += 1
    elif i == 2:
        cnt[2] += 1
    elif i == 3:
        cnt[3] += 1


for i in range(len(cnt)):
    price = prices[i]*cnt[i]
    all_prices += price
    print(f'{names[i]} {cnt[i]}개 이고, 가격은 {price}원 입니다.')
    print()


print(f'총 가격은 {all_prices}원 이다.')

'''