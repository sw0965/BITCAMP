from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

#다중분류 모델 
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)  #506, 13
print(y.shape)  #506,

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# n_estimateors = 100
# learning_rate = 0.7
# colsample_bytree = 0.11
# colsample_bylevel = 0.1

# max_depth = 8
# n_jobs = -1

# model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimateors,
                    #   n_jobs=n_jobs, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel)
# model = XGBClassifier(n_estimators=100, learning_rate=0.1, colsample_bytree=0.4, colsample_bylevel=0.5, max_depth=4, n_jobs=-1)
model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(' 점수 : ', score)

print(model.feature_importances_)


#  점수 :  0.9
