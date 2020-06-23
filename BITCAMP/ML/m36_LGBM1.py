'''
1. 회귀
2. 이진분류
3. 다중 분류

1. eval 에 'loss' 와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. Plot 으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시.

5. m27 ~ m29 까지 왁벽 이해할 것.
'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

# dataset = load_boston()
x, y = load_boston(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state=66, train_size=0.8)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

Parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree": [0.6, 0.9, 1]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6, 0.7, 0.9]}
]

# model = GridSearchCV(XGBRegressor(), Parameters, cv=5, n_jobs=-1)
model = LGBMRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(' 점수 : ', score)
# print('______________________________________________________________________')
# print(model.best_estimator_)
# print('______________________________________________________________________')
# print(model.best_params_)
# print('______________________________________________________________________')

print(model.feature_importances_)


thresholds = np.sort(model.feature_importances_)
print(thresholds)




for thresh in thresholds:  # 칼럼 수 만큼 돈다!
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    # select_y_train = selection.transform(y_train)
    # print(select_x_train.shape)
    # print(type(select_x_train))
    # print(type(y_train))
    selection_model = LGBMRegressor(n_estimators=1, n_jobs=-1)

    selection_model.fit(select_x_train, y_train, verbose=True, eval_metric=['rmse','logloss'], eval_set=[(select_x_train, y_train),(select_x_test, y_test)],
            early_stopping_rounds=100)


    # results = selection_model.evals_result()
    # print("eval's result: ", results)   

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


        