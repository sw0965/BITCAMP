import numpy as np
x1 = np.array([range(1,101),range(311,411),range(100)])
y1 = np.array(range(711,811))



x1 = np.transpose(x1)
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split( 
x1, y1, shuffle=True, train_size=0.8)



from keras.models import Sequential, Model
from keras.layers import Dense, Input 

input1 = Input(shape=(3, ))
dense1_1 = Dense(10, activation='relu', name='ait1')(input1)
dense1_2 = Dense(20, activation='relu', name='ait2')(dense1_1)
dense1_3 = Dense(30, activation='relu', name='ait3')(dense1_2)


output1 = Dense(30, name='ot1_1')(dense1_3) 
output1_2 = Dense(10, name='ot1_2')(output1)
output1_3 = Dense(8, name='ot1_3')(output1_2)
output1_4 = Dense(1, name='ot1_4')(output1_3)


model = Model(inputs = input1, outputs = output1_4)


model.compile(loss='mse',optimizer='adam', metrics=['mse']) 

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') 
model.fit(x1_train,y1_train,epochs=87,batch_size=1,validation_split=0.25, verbose=3) #, callbacks=[early_stopping])


loss = model.evaluate(x1_test,y1_test,batch_size=1)
print("loss : ",loss)


y1_predict = model.predict(x1_test)


from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE = RMSE(y1_test, y1_predict)
print("RMSE : ", RMSE)



from sklearn.metrics import r2_score
r2 = r2_score(y1_test,y1_predict)
print("R2 : ", r2)