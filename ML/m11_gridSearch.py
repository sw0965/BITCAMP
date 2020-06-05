import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC




#1. 데이터
iris = pd.read_csv('./data/csv/study/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

# print(x)
# print(y)

#. 트레인 테스트 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

parameters =[
    {"C": [1, 10, 100, 1000], "kernel":["linear"]},
    {"C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]},
]

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters, cv=kfold)# cv= 5라고 써도됌 

model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
''' 내 파라미터중 제일 좋은거 찾았을때 밑에 나온게 결괏값
최적의 매개변수 :  SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
'''

y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test, y_pred)) # 뭐가 acc : 1이 나온지 모름

# kfold 먼저쓰고 train test 를 써도되고 반대로 써도됨 train test 를 분류하고 train 에서 kfold=5니까 train중 80프로는 train이 한번 더 돌고
#나머지 20프로는 val로 돌아간다 (훈련이 잘 되는 과정인듯)


#grid 내가 넣어놓은 모든걸 싹쓸이 하는 것
# 그렇지만싹 가져가는게 좋지않고  dropout처럼  몇개를 떨궈야됌