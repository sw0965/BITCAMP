from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

#다중분류 모델 
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)  #506, 13
print(y.shape)  #506,

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

n_estimateors = 100
learning_rate = 0.7
colsample_bytree = 0.11
colsample_bylevel = 0.1

max_depth = 8
n_jobs = -1

Parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree": [0.6, 0.9, 1]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6, 0.7, 0.9]}
]


model = GridSearchCV(XGBRegressor(), Parameters, cv=5, n_jobs=-1)
model.fit(x_train, y_train)
print('______________________________________________________________________')
print(model.best_estimator_)
print('______________________________________________________________________')
print(model.best_params_)
print('______________________________________________________________________')



score = model.score(x_test, y_test)
print(' 점수 : ', score)
# 점수 :  0.9368795067163034