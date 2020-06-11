import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

########## 데이터 불러오기 ##########
x         = np.load('./data/dacon/KAERI/x_train_pipe.npy', allow_pickle='ture')
y         = np.load('./data/dacon/KAERI/x_test_pipe.npy', allow_pickle='ture')
x_predict = np.load('./data/dacon/KAERI/x_pred_pipe.npy', allow_pickle='ture')

print(x.shape)           # 2800 375 4
print(y.shape)           # 2800 4
print(x_predict.shape)   # 700 375 4


########## 머신러닝 사용을 위해 2차원으로 바꿔주기 ##########
x = x.reshape(2800, 1500)
x_predict = x_predict.reshape(700, 1500)


########## train test 분리 ##########
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=43)



########## 그리드/랜덤 서치에서 사용할 매개 변수 ##########
parameters = [{'R__n_estimators' : [100, 100],'R__min_samples_split':[10, 100], 
            'R__max_leaf_nodes' :[10, 100], 'R__max_depth' : [10, 100],'R__min_samples_leaf' : [10, 100]}]


########## pipe 모델 구성 ##########
pipe = Pipeline([("scaler", StandardScaler()), ('R', RandomForestRegressor())])


########## RandomizedSearchCV 사용 ##########
model = RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=-1)


########## 훈련 ##########
model.fit(x_train, y_train)


########## 평가, 예측 ##########
mse = model.score(x_test, y_test)
print("최적의 매개 변수 : ", model.best_estimator_)
print("mse : ", mse)   
'''mse :  0.9805900401035631'''


########## predict 해주기 ##########
y_pred = model.predict(x_predict)
print(y_pred)   
'''
[[-222.8583922   -70.40912785   90.63120722    0.64402125]
 [ 325.54800307 -316.06515155   96.32415987    0.5593598 ]
 [  -9.10667805  246.50986825   93.48649336    0.56287124]
 ...
 [ 370.58147443 -309.5017011    95.66553269    0.58066859]
 [ 254.47203075 -339.44414432   98.8275972     0.54598364]
 [ 203.33288784  271.44706351   97.04333119    0.57824797]]
'''

########## 서브미션 csv파일 저장 ##########
# submissions = pd.DataFrame({
#     "id": range(2800,3500),
#     "X": y_pred[:,0],
#     "Y": y_pred[:,1],
#     "M": y_pred[:,2],
#     "V": y_pred[:,3]
# })

# submissions.to_csv('./comp3_sub.csv', index = False)