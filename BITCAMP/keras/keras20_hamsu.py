#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).transpose()
y = np.array(range(711,811)).transpose()

print(x.shape)


from sklearn.model_selection import train_test_split
# train_size=0.9로 잡으면 1,3번째 변수에 0.95만큼 나머지 자동 test_size를 잡아주면 2,4번째로 할당하고 나머지 자동 
x_train,x_test,y_train,y_test = train_test_split( 
    # x,y,random_state = 66, shuffle=True,
    x, y, shuffle=False, train_size=0.8)
    # train_size=0.8, test_size=0.1    0.1은 그냥 버려짐 
    # # test_size=0.05 둘중 하나만 쓰면 나머지는 알아서 자동으로 잡히는거 같은데..
    

print("x_train",x_train,"\ny_train",y_train)
print("x_test",x_test,"\ny_test",y_test)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input # DNN구조의 기본
# model = Sequential()
# model.add(Dense(5,input_dim=3))#인풋 1개 첫 아웃풋5개 activation도 default가 있음
# model.add(Dense(4))
# model.add(Dense(1))


input1 = Input(shape=(3, ))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(80, activation='relu')(dense1)
dense1 = Dense(70, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(30, activation='relu')(dense1)
output1 = Dense(1)(dense1)


input1 = Input(shape=(3, ))
dense1 = Dense(222, activation='relu')(input1)
dense2 = Dense(222, activation='relu')(dense1)
dense3 = Dense(222, activation='relu')(dense2)
dense4 = Dense(222, activation='relu')(dense3)
dense5 = Dense(111, activation='relu')(dense4)
dense6 = Dense(100, activation='relu')(dense5)
dense7 = Dense(100, activation='relu')(dense6)
dense8 = Dense(50, activation='relu')(dense7)
dense9 = Dense(30, activation='relu')(dense8)
dense10 = Dense(10, activation='relu')(dense9)
output1 = Dense(1)(dense10)

model = Model(inputs = input1, outputs = output1) #히든레이어 명시가 필요없으니 위에 명시를 해주는것 Sequential() 이거처럼

model.summary()


# 0.019=100, 0.007 0.9935

#3. 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mse']) # 회기방식과 분류방식 2가지 ?  # mse는 실제 값과 예측값의 차이를 평균하는것 
model.fit(x_train,y_train,epochs=600, batch_size=1,
            validation_split=0.25, verbose=3) # batch_size = 32(default)

model.summary()

#4. 평가, 예측
loss,mse = model.evaluate(x_test,y_test,batch_size=1) # evaluate -> 결과 반환(기본적으로 loss와 metrics를 반환)을 loss와 acc에 받겠다.

print("loss : ",loss)
print("mse : ",mse)


y_predict = model.predict(x_test)
print(y_test)
print(y_predict)


#RMSE 구하기 #낮을수록 좋다
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test,y_predict))


#R2 구하기 # 1에 근접할수록 좋다. 다른 보조지표와 같이 쓴다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("r2 : ",r2_y_predict)



 # Question


 # Note
# verbose 넣어보기

 # homework

