import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, plot_importance, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt


x, y = load_iris(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state=66, train_size=0.8)

print(x_train.shape)  #120 4
print(x_test.shape)   #30 4
print(y_train.shape)  #120 
print(y_test.shape)   #30

model = XGBClassifier(objective='multi:softmax', n_estimators=100, learning_rate=0.1, n_jobs=-1)


model.fit(x_train, y_train, verbose=True, eval_metric=['auc','mlogloss'], eval_set=[(x_train, y_train),(x_test, y_test)],
            early_stopping_rounds=100)


results = model.evals_result()
print("eval's result: ", results)


y_pred = model.predict(x_test)
print(y_pred.shape) #114,

score = accuracy_score(y_pred, y_test)
# print('r2 Score : %.2f%%'%(r2 *100))
print('점수 : ', score)


# import matplotlib.pyplot as plts

# epochs = len(results['validation_0']['mlogloss'])
# x_axis = range(0,epochs)


# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['mlogloss'], label = 'Train')
# ax.plot(x_axis, results['validation_1']['mlogloss'], label = 'Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# # plt.show()



# epochs = len(results['validation_0']['merror'])
# x_axis = range(0, epochs)


# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['merror'], label = 'Train')
# ax.plot(x_axis, results['validation_1']['merror'], label = 'Test')
# ax.legend()
# plt.ylabel('ERROR')
# plt.title('XGBoost Error')
# plt.show()