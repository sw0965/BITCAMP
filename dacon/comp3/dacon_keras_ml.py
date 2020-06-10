import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#1. 데이터

x         = np.load('./data/dacon/KAERI/x_train_pipe.npy', allow_pickle='ture')
y         = np.load('./data/dacon/KAERI/x_test_pipe.npy', allow_pickle='ture')
x_predict = np.load('./data/dacon/KAERI/x_pred_pipe.npy', allow_pickle='ture')

x = x.reshape(2800, 1500)
x_predict = x_predict.reshape(700, 1500)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)

print(x_train.shape)   #2240, 1500
print(x_test.shape)    #560, 1500
print(y_train.shape)   #2240, 4
print(y_test.shape)    #560, 4


def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(1500, ), name='input')
    x = Dense(64, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, name= 'hidden2')(x)
    x = Dense(256, name= 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(128, name= 'hidden4')(x)
    outputs = Dense(4, name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Parameter 변수
def create_hyper_parameters():
    batches = [121, 122, 124,126]
    optimizers = ['adam']
    dropout = [0.1]
    epochs = [95, 100, 105]
    return{"model__batch_size" : batches, "model__optimizer" : optimizers, 
           "model__drop" : dropout, 'model__epochs': epochs}

model = KerasRegressor(build_fn = build_model, verbose = 1)  # sklearn 에서 쓸 수 있게 랩핑을 함
parameter = create_hyper_parameters()

pipe = Pipeline([('scaler',StandardScaler()), ('model',model)])

search = RandomizedSearchCV(pipe, parameter, cv=3)

search.fit(x_train, y_train)

mse = search.score(x_test, y_test)
print(search.best_params_)
print("최종 정답률 : ", mse)

y_pred1 = search.predict(x_test)

def kaeri_metric(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_test, y_pred1) + 0.5 * E2(y_test, y_pred1)


### E1과 E2는 아래에 정의됨 ###

def E1(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_test)[:,:2], np.array(y_pred1)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_test)[:,2:], np.array(y_pred1)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
print('E1 : ', E1)
print('E2 : ', E2)


y_pred = search.predict(x_predict)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./comp3_tod1.csv', index = False)



# 에러코드
#The model is not configured to compute accuracy. You should pass `metrics=["accuracy"]` to the `model.compile()` method.


#monmax
#{'model__optimizer': 'adam', 'model__epochs': 100, 'model__drop': 0.1, 'model__batch_size': 128}
# 최종 정답률 :  -17628.025948660714

#minmax 1차수정
# {'model__optimizer': 'adam', 'model__epochs': 150, 'model__drop': 0.1, 'model__batch_size': 100}
# 최종 정답률 :  -17129.114711216516

# minmax 2차수정
# {'model__optimizer': 'adam', 'model__epochs': 100, 'model__drop': 0.1, 'model__batch_size': 128}
# 최종 정답률 :  -17593.721372767857

# standard
# {'model__optimizer': 'adam', 'model__epochs': 100, 'model__drop': 0.1, 'model__batch_size': 120}
# 최종 정답률 :  -14899.37095424107

# standard 1차
# {'model__optimizer': 'adam', 'model__epochs': 100, 'model__drop': 0.1, 'model__batch_size': 122}
# 최종 정답률 :  -14643.133276367187 

# stand 2차
# {'model__optimizer': 'adam', 'model__epochs': 95, 'model__drop': 0.1, 'model__batch_size': 122}
# 최종 정답률 :  -14818.28850795201

# stand 3차
# {'model__optimizer': 'adam', 'model__epochs': 95, 'model__drop': 0.1, 'model__batch_size': 126}
# 최종 정답률 :  -14376.622875976562