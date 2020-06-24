import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./DATA/dacon/bio/train.csv', index_col=0)
test  = pd.read_csv('./DATA/dacon/bio/test.csv', index_col=0)

print(train.shape)
print(test.shape)




# print(train.info())
train_drop_na = train.dropna()
# print(train_drop_na)
train_drop_na.T.plot()
# plt.show()

rho_src = train_drop_na.iloc[:, :36]
target = train_drop_na.iloc[:, 71:]
y = train_drop_na.iloc[:,36:71]
# print(x1)
# print(x2)
print(y.shape)  # (3, 35)

x = pd.concat([rho_src, target], axis=1)
print(x.shape)  # (3, 40)


# test = train.drop([575, 3609, 6897])
# print(train_for_dst)

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

x_train, x_test, y_test, y_train = train_test_split(x, y, random_state=44, shuffle=True, test_size=0.2)


model = Sequential()

model.add(Dense(64, input_dim=40))
model.add(Dense(128))
model.add(Dense(356))
model.add(Dense(712))
model.add(Dense(356))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(35))

model.summary()

model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train,epochs=140, validation_data=0.2)

# loss,mse = model.evaluate(x_test,y_test)
# print('loss :', loss)
# print('mse :', mse)

# y_predict = model.predict(x_test)

# loss,mse = model.evaluate(y_test, y_predict)
# print('loss :', loss)
# print('mse :', mse)













'''

train_dst = train.iloc[:,36:71]
test_dst = test.iloc[:,36:]
y = train.iloc[:,71:]

# print(train_dst.notnull())
train_dst_not = train_dst.dropna()



train_dst = train_dst.interpolate(methods = 'linear', axis= 1)
test_dst = test_dst.interpolate(methods = 'linear', axis= 1)


# print(train_dst.info())
# print(test_dst.info())





train_650dst = train_dst.iloc[:,0]
train_660dst = train_dst.iloc[:,1]
train_670dst = train_dst.iloc[:,2]
train_680dst = train_dst.iloc[:,3]
train_690dst = train_dst.iloc[:,4]
train_700dst = train_dst.iloc[:,5]


# print(train_dst.iloc[:,0])
# print(train_dst.notnull())
'''