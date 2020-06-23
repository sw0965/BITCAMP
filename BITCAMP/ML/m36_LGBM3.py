import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, plot_importance, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier


# dataset = load_boston()
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state=66, train_size=0.8)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LGBMClassifier(objective='multiclass')

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(' 점수 : ', score)



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
    selection_model = LGBMClassifier(n_estimators=5, n_jobs=-1)

    selection_model.fit(select_x_train, y_train, verbose=True, eval_metric=['multi_logloss', 'multi_error'] ,eval_set=[(select_x_train, y_train),(select_x_test, y_test)],
            early_stopping_rounds=100)


    # results = selection_model.evals_result()
    # print("eval's result: ", results)   

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


# model.save_model("./MODEL/sample/xgb_save/best_iris.xgb.model")