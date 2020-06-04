import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.layers import Dense
from keras.models import Sequential

##################################################################################################
####################################### keras로 변형해보기 ########################################
##################################################################################################

#1. 데이터  
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])
print(x_data)
print(y_data)
print(x_data.shape) #(4, 2)
print(y_data.shape) #(4, )




#2. 모델
model = Sequential()
model.add(Dense(2, input_dim=(2)))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100)

#4. 평가
loss, acc = model.evaluate(x_data, y_data)
print('loss : ', loss)
print('acc : ', acc)


x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)
print(y_predict)
print(y_predict.shape)
y_predict = np.transpose(y_predict)


# for i in y_predict:
#     i = y_predict(:4):
#     if i >= 0.5:
#         1
#     else: 0
# print(y_predict)

# for i in y_predict:            
#         predict = y_predict[i]         
#         if predict >= 0.5:
#             1
#         else:
#             0
# print(y_predict)