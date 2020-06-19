import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestClassifier
from   sklearn.datasets         import load_breast_cancer





#1. 데이터
dataset = load_breast_cancer()

x       = dataset.data
y       = dataset.target

print(x.shape)  #536, 30
print(y.shape)  #569,

#. 트레인 테스트 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
#, 'min_samples_leaf' : [8, 12, 18], 'min_samples_split' : [8, 16, 20]
parameters = {'n_estimators' : [10], 'max_depth' : [1, 10, 100, 1000],'max_features':['auto'],
'max_leaf_nodes':[None], 'class_weight':[None], 'criterion':['gini'],
'min_impurity_decrease':[0.0], 'min_impurity_split' : [None],
'min_samples_leaf':[1], 'min_samples_split':[2], 'warm_start':[False],
'min_weight_fraction_leaf': [0.0],'bootstrap':[True], 'n_jobs':[None],
'oob_score':[False], 'random_state':[None], 'verbose':[0]}
    

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1)# cv= 5라고 써도됌 

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)

''' 
#내 파라미터중 제일 좋은거 찾았을때 밑에 나온게 결괏값
'''


y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred)) # 뭐가 acc : 1이 나온지 모름
'''최종 정답률 :  0.9912280701754386'''

# kfold 먼저쓰고 train test 를 써도되고 반대로 써도됨 train test 를 분류하고 train 에서 kfold=5니까 train중 80프로는 train이 한번 더 돌고
#나머지 20프로는 val로 돌아간다 (훈련이 잘 되는 과정인듯)


#grid 내가 넣어놓은 모든걸 싹쓸이 하는 것
# 그렇지만싹 가져가는게 좋지않고  dropout처럼  몇개를 떨궈야됌
