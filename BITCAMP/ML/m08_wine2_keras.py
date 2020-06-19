import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils              import np_utils


##################################################################################################
####################################### keras로 변형해보기 ########################################
##################################################################################################

#1. 데이터  
x = np.load('./data/x_wine.npy', allow_pickle='ture')
y = np.load('./data/y_wine.npy', allow_pickle='ture')
print(x.shape)  #(4898, 9)
print(y.shape)  #(4898, )



# 트레인 테스트 스플릿
x_train,x_test,y_train,y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('ytr.shape :', y_train.shape)   #(3918, 10)
print('yte.shape :', y_test.shape)    #(980, 10)


# 전처리
# scaler = MinMaxScaler()
# scaler.fit(x_train)   
# x_train = scaler.transform(x_train)
# x_test  = scaler.transform(x_test)

scaler = StandardScaler()
scaler.fit(x_train)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)



print(x_train.shape)   #(3918, 9)
print(x_test.shape)    #(980, 9)
print(y_train.shape)   #(3918, 10)
print(y_test.shape)    #(980, 10)


#2. 모델
model = Sequential()
model.add(Dense(2, input_dim=(9)))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=8)

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss : ', loss)
print('acc : ', acc)

