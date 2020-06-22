import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRFClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state=66, train_size=0.8)

model = XGBRFClassifier(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric='error', eval_set=[(x_train, y_train),(x_test, y_test)])

results = model.evals_result()
print("eval's result: ", results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('acc :', acc)

import pickle # 파이썬 제공
pickle.dump(model,open("./MODEL/sample/xgb_save/cancer.pickle.dat", "wb"))

print('저장')

model2 = pickle.load(open("./MODEL/sample/xgb_save/cancer.pickle.dat", 'rb'))
print('로드')

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('acc :', acc)