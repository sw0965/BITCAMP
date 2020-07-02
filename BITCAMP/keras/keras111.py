import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, Dense, MaxPooling2D, LSTM

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255
x_train = x_train.reshape(x_train.shape[0], 28*28)/255
x_test = x_test.reshape(x_test.shape[0], 28*28)/255
print(x_train.shape) # (60000, 28, 28, 1)
print(x_test.shape)  # (10000, 28, 28, 1)



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)   
print(y_train.shape)
print(y_test.shape)

#2. 모델   (모델 자체를 함수로 만든다.) 

# Model 변수
def build_model(dropout, optimizer, activation, learning_rate):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(32, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(256, activation=activation, name= 'hidden2')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation=activation, name= 'hidden3')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(10, activation=activation, name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    return model

# Parameter 변수
from keras.optimizers import Adam, RMSprop
from keras.activations import relu,selu,elu,softmax
def create_hyper_parameters():
    batches = [10, 20, 30]
    optimizers = [RMSprop, Adam]
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    learning_rate = [0.001, 0.01]
    activation = [relu, elu, softmax]
    return{"batch_size" : batches, "optimizer" : optimizers, "dropout" : dropout, "learning_rate" : learning_rate, "activation" : activation}


# 바로 그리드 서치에 적용이 안되기 때문에 wrap 을 땡겨온다

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

model = KerasClassifier(build_fn = build_model)  # sklearn 에서 쓸 수 있게 랩핑을 함

hyperparameters = create_hyper_parameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
acc = search.score(x_test, y_test)

print(search.best_params_)
print("최종 정답률 : ", acc)

# {'optimizer': <class 'keras.optimizers.Adam'>, 'learning_rate': 0.001, 'dropout': 0.4, 'batch_size': 20, 'activation': <function relu at 0x000002342A162438>}
# 최종 정답률 :  0.7838000059127808