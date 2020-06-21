import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, plot_importance, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

# dataset = load_boston()
x, y = load_iris(return_X_y=True)

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


# model = GridSearchCV(XGBClassifier(), Parameters, cv = 5, n_jobs=-1)
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print(' 점수 : ', score)

# print('______________________________________________________________________')
# print(model.best_estimator_)
# print('______________________________________________________________________')
# print(model.best_params_)
# print('______________________________________________________________________')

'''
 점수 :  0.9333333333333333
______________________________________________________________________
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
'''

# model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#               colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=200, n_jobs=0, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)

model = XGBClassifier()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(' 점수 : ', score)

# 그리드 서치
#  점수 :  0.9333333333333333
# 디폴트
#  점수 :  0.9
print(model.feature_importances_)

thresholds = np.sort(model.feature_importances_)
print(thresholds)



plot_importance(model)
# plt.show()










for thresh in thresholds:  # 칼럼 수 만큼 돈다!
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                     # threshold= median

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = GridSearchCV(XGBClassifier(), Parameters, cv = 5, n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    print('______________________________________________________________________')
    print(selection_model.best_estimator_)
    print('______________________________________________________________________')
    print(selection_model.best_params_)
    print('______________________________________________________________________')

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
'''
# 그리드서치
(120, 4)
Thresh=0.087, n=4, R2: 84.90%
(120, 3)
Thresh=0.179, n=3, R2: 89.93%
(120, 2)
Thresh=0.363, n=2, R2: 94.97%
(120, 1)
Thresh=0.372, n=1, R2: 89.93%

# 디폴트
(120, 4)
Thresh=0.018, n=4, R2: 84.90%
(120, 3)
Thresh=0.026, n=3, R2: 84.90%
(120, 2)
Thresh=0.337, n=2, R2: 94.97%
(120, 1)
Thresh=0.619, n=1, R2: 89.93%
'''
'''
(120, 4)
______________________________________________________________________
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
______________________________________________________________________
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200}
______________________________________________________________________
Thresh=0.018, n=4, R2: 89.93%
(120, 3)
______________________________________________________________________
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=300, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
______________________________________________________________________
{'colsample_bylevel': 0.6, 'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300}
______________________________________________________________________
Thresh=0.026, n=3, R2: 94.97%
(120, 2)
______________________________________________________________________
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
______________________________________________________________________
{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
______________________________________________________________________
Thresh=0.337, n=2, R2: 94.97%
(120, 1)
______________________________________________________________________
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
______________________________________________________________________
{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
______________________________________________________________________
Thresh=0.619, n=1, R2: 89.93%
'''