import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#1. 데이터

x         = np.load('./data/dacon/KAERI/x_train_pipe.npy', allow_pickle='ture')
y         = np.load('./data/dacon/KAERI/x_test_pipe.npy', allow_pickle='ture')
x_predict = np.load('./data/dacon/KAERI/x_pred_pipe.npy', allow_pickle='ture')

print(x.shape)           # 2800 375 4
print(y.shape)           # 2800 4
print(x_predict.shape)   # 700 375 4

x = x.reshape(2800, 1500)
x_predict = x_predict.reshape(262500, 4)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)



# 그리드/랜덤 서치에서 사용할 매개 변수
parameters = [{'R__n_estimators' : [100, 200],'R__min_samples_split':[10, 100], 
            'R__max_leaf_nodes' :[10, 100], 'R__max_depth' : [10, 100],'R__min_samples_leaf' : [10, 100]}]


#2. 모델

pipe = Pipeline([("scaler", StandardScaler()), ('R', RandomForestRegressor())])

model = RandomizedSearchCV(pipe, parameters, cv=5)

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
mse = model.score(x_test, y_test)
print("최적의 매개 변수 : ", model.best_estimator_)
print("mse : ", mse)


y_pred = model.predict(x_predict)
print(y_pred)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./comp3_sub.csv', index = False)