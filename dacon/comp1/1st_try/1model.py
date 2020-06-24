import numpy as np
import pandas as pd
from sklearn.model_selection  import  train_test_split
from xgboost import XGBRegressor, plot_importance ,XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

train     = np.load('./DACON//comp1/1st_try/data/train.npy', allow_pickle='ture')
x_predict = np.load('./DACON//comp1/1st_try/data/test.npy', allow_pickle='ture')
y_predict = np.load('./DACON//comp1/1st_try/data/y_predict.npy', allow_pickle='ture')

# print(train)      # (10000, 75)
# print(x_predict)      #  (10000, 71)
# print(y_predict)  # (10000, 4)

# print(x_predict.shape)        
# print(y_predict.shape)   

rho = train[:,:1]
print(rho)
src = train[:,1:36]
dst = train[:,36:71]
y = train[:,71:]
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 44, shuffle= True)

# print(x_train.shape)  (8000, 71)
# print(x_test.shape)   (2000, 71)
# print(y_train.shape)  (8000, 4)
# print(y_test.shape)   (2000, 4)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)



model = Sequential()

model.add(Dense(32, input_shape=(71,)))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(356))
# model.add(Dense(812))
model.add(Dense(356))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, verbose=2, validation_split=0.2)

loss, mse = model.evaluate(x_test, y_test)
print('ls : ', loss)
print('mse : ', mse)



y_pred = model.predict(x_predict)
print(y_pred)




# model = XGBRegressor()
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print(' 점수 : ', score)

# print(model.feature_importances_)
# plot_importance(model)
# plt.show()

# model = GridSearchCV(XGBClassifier(), Parameters, cv = 5, n_jobs=-1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print(' 점수 : ', score)
'''



# a = np.arange(10000,20000)
# submission = pd.DataFrame(y_predict, a)
# submission.to_csv(index = True, header=['hhb','hbo2','ca','na'],index_label='id')