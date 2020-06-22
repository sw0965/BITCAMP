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

model = XGBRegressor()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(' 점수 : ', score)



# print('______________________________________________________________________')
# print(model.best_estimator_)
# print('______________________________________________________________________')
# print(model.best_params_)
# print('______________________________________________________________________')
# score = model.score(x_test, y_test)
# print('R2 : ', score)

# print(model.feature_importances_)

# plot_importance(model)
# # plt.show()

thresholds = np.sort(model.feature_importances_)
print(thresholds)

import time
start = time.time()

for thresh in thresholds:  # 칼럼 수 만큼 돈다!
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                     # threshold= median

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

end = time.time() - start
print(end)



import time
start2 = time.time()

for thresh in thresholds:  # 칼럼 수 만큼 돈다!
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                     # threshold= median

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

end2 = time.time() - start2
print(end2)
