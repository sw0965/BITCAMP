import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1. 데이터
iris = load_iris()
x = iris.data
y = iris.target
print(x.shape)  #(150, 4)
print(y.shape)  #(150, )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)

#2. 모델
# model = SVC()  
# svc_model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC()) # make_pipeline 이건 전처리랑 모델만 써주면

pipe.fit(x_train, y_train)

print("acc: ", pipe.score(x_test, y_test))


# acc:  0.9666666666666667