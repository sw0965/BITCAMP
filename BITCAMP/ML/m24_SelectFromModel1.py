import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

# dataset = load_boston()
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state=66, train_size=0.8)


Parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree": [0.6, 0.9, 1]},
    {"n_estimators" : [100, 200, 300], "learning_rate":[0.1, 0.3, 0.5, 0.01],
    "max_depth": [4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6, 0.7, 0.9]}
]


# model = GridSearchCV(XGBRegressor(), Parameters  ,cv = 5, n_jobs=-1)
# print('______________________________________________________________________')
# print(model.best_estimator_)
# print('______________________________________________________________________')
# print(model.best_params_)
# print('______________________________________________________________________')

# score = model.score(x_test, y_test)
# print(' 점수 : ', score)
'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
#  점수 :  0.9368795067163034 
'''

model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
model.fit(x_train, y_train)
# 점수 :  0.9368795067163034
# 점수 :  0.9284598140318608


score = model.score(x_test, y_test)
print(' 점수 : ', score)

# score = model.score(x_test, y_test)
# print('R2 : ', score)
print(model.feature_importances_)

plot_importance(model)
# plt.show()

thresholds = np.sort(model.feature_importances_)
print(thresholds)




for thresh in thresholds:  # 칼럼 수 만큼 돈다!
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                     # threshold= median

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = GridSearchCV(XGBRegressor(), Parameters, n_jobs=-1, cv=5)
    # print('______________________________________________________________________')
    # print(selection_model.best_estimator_)
    print('______________________________________________________________________')
    # print(selection_model.best_params_)
    # print('______________________________________________________________________')
    # selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
'''
# 그리드 서치까지 묶기, SelectFromModel 파라미터 알아보기 (숙제)
'''
'''
R2 :  0.9221188544655419
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
(404, 13)
Thresh=0.001, n=13, R2: 92.21%
(404, 12)
Thresh=0.004, n=12, R2: 92.16%
(404, 11)
Thresh=0.012, n=11, R2: 92.03%
(404, 10)
Thresh=0.012, n=10, R2: 92.19%
(404, 9)
Thresh=0.014, n=9, R2: 93.08%
(404, 8)
Thresh=0.015, n=8, R2: 92.37%
(404, 7)
Thresh=0.018, n=7, R2: 91.48%
(404, 6)
Thresh=0.030, n=6, R2: 92.71%
(404, 5)
Thresh=0.042, n=5, R2: 91.74%
(404, 4)
Thresh=0.052, n=4, R2: 92.11%
(404, 3)
Thresh=0.069, n=3, R2: 92.52%
(404, 2)
Thresh=0.301, n=2, R2: 69.41%
(404, 1)
Thresh=0.428, n=1, R2: 44.98%
'''