import numpy as np
from keras.models import load_model, Sequential
import matplotlib.pyplot as plt

# 데이터
x_train = np.load('./mini_project/npy_data/x_train_test1.npy')
x_test = np.load('./mini_project/npy_data/x_test_test1.npy')
y_train = np.load('./mini_project/npy_data/y_train_test1.npy')
y_test = np.load('./mini_project/npy_data/y_test_test1.npy')
x_predict = np.load('./mini_project/npy_data/x_predict_test1.npy')


# 모델
model = load_model('./mini_project/DONE/best.h5')

# 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# 시각화


'''
# print(x_predict)
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict)
print('')



for i in y_predict:
    if i == 0:
        print('사과 입니다')
        print('')
    elif i == 1:
        print('바나나 입니다') 
        print('')
    elif i == 2:
        print('포도 입니다.')
        print('')
    elif i == 3:
        print('파인애플 입니다.')
        print('')
'''