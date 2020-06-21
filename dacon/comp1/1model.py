import numpy as np
import pandas as pd
from sklearn.model_selection  import  train_test_split
from xgboost import XGBRegressor, plot_importance ,XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
train      = pd.read_csv('./DACON/comp1/data/train_fix.csv',index_col=0, header=0)
x_predict  = pd.read_csv('./DACON/comp1/data/test_fix.csv', index_col=0, header=0)
y_predict  = pd.read_csv('./DATA/dacon/bio/sample_submission.csv', index_col=0, header=0)

# print(train.shape)       (10000, 75)
# print(test.shape)        (10000, 71)
# print(y_predict.shape)   (10000, 4)

x = train.iloc[:,:71]
y = train.iloc[:,71:]
# print(x)
# print(y)

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


model = XGBRegressor()
model.fit(x_train, y_train)



# score = model.score(x_test, y_test)
# print(' 점수 : ', score)

# print(model.feature_importances_)
# plot_importance(model)
# plt.show()

# model = GridSearchCV(XGBClassifier(), Parameters, cv = 5, n_jobs=-1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print(' 점수 : ', score)