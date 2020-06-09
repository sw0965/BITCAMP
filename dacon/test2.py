import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

############### 데이터 불러오기 ###############
train      = pd.read_csv('./data/dacon/bio/train.csv',index_col=0, header=0)
test       = pd.read_csv('./data/dacon/bio/test.csv', index_col=0, header=0)
submission = pd.read_csv('./data/dacon/bio/sample_submission.csv', index_col=0, header=0)

# print('train.shape : ', train.shape)            #(10000, 75)
# print('test.shape : ', test.shape)              #(10000, 71)  = x_predict
# print('submission.shape : ', submission.shape)  #(10000, 4)   = y_predict


############### 데이터 보관 ###############

train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 
train = train.interpolate() 
test = test.interpolate()

############### numpy 전환 ###############

train      = train.values
test       = test.values
submission = submission.values

############### 슬라이싱 ###############

x = train[:,:71]
y = train[:,71:]

############### 트레인 테스트 분리 ###############

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size = 0.2)
print(x_train.shape) #(8000, 71)
print(x_test.shape)  #(8000, 4)
print(y_train.shape) #(2000, 71)
print(y_test.shape)  #(2000, 4)

scaler = StandardScaler()
scaler.fit(x)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)


#. 모델구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(71,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(drop)(x)
    x = Dense(356, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(712, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(356, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(drop)(x)
    outputs = Dense(4)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model


def create_hyper_parameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score

model = KerasRegressor(build_fn = build_model)  # sklearn 에서 쓸 수 있게 랩핑을 함

hyperparameters = create_hyper_parameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
r2 = search.score(x_test, y_test)

print(search.best_params_)
print("최종 정답률 : ", r2)



# # model.summary()
# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
# model.compile(loss='mae', optimizer='adam', metrics=['mae'])
# model.fit(x_train, y_train, epochs=125, callbacks=[early_stopping])
# loss, mae = model.evaluate(x_test, y_test) 
# print("loss : ", loss)
# print('mae : ', mae)

# submission = model.predict(test)
# print(submission.shape) #(10000, 4)

# print(submission)

# print(type(submission))

# submission = pd.DataFrame(submission)
# print(type(submission))


# loss, mae = model.evaluate(test, submission) 
# print("loss : ", loss)
# print('mae : ', mae)



