# 과적합 방지
#1. 훈련데이터량을 늘린다.
#2. 피쳐수를 줄인다.
#3. regularliztion

from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

#회귀 모델 
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

# model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimateors,
                    #   n_jobs=n_jobs, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, colsample_bytree=0.4, colsample_bylevel=0.5, max_depth=4, n_jobs=-1)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(' 점수 : ', score)
#  점수 :  0.9335151588615975
print(model.feature_importances_)





plot_importance(model)
# plt.show()