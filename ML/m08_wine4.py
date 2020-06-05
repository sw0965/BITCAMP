import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/wine/winequality-white.csv', index_col = 0, header=0, sep=';', encoding='CP949')

# 판다스 슬라이싱
x = wine.drop('quality', axis=1)
y = wine['quality']

print(x.shape)
print(y.shape)


# Y 레이블 축소
# 3~9 인 퀄리티를 0, 1, 2로 축소     # 0 , 1 , 2
newlist = []
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc_score : ', accuracy_score(y_test, y_pred))
print('acc       : ', acc)