import numpy as np
import pandas as pd
from pandas import DataFrame

wine = pd.read_csv('./data/csv/wine/winequality-white.csv', index_col = 0, header=0, sep=';', encoding='CP949')

print(wine)     
print(wine.shape)   #(4898, 11)

'''
wine.info()
<class 'pandas.core.frame.DataFrame'>
Float64Index: 4898 entries, 7.0 to 6.0
Data columns (total 11 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   volatile acidity      4898 non-null   float64
 1   citric acid           4898 non-null   float64
 2   residual sugar        4898 non-null   float64
 3   chlorides             4898 non-null   float64
 4   free sulfur dioxide   4898 non-null   float64
 5   total sulfur dioxide  4898 non-null   float64
 6   density               4898 non-null   float64
 7   pH                    4898 non-null   float64
 8   sulphates             4898 non-null   float64
 9   alcohol               4898 non-null   float64
 10  quality               4898 non-null   int64
dtypes: float64(10), int64(1)
'''

'''
print(wine.head())
               volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
fixed acidity
7.0                        0.27         0.36            20.7      0.045                 45.0                 170.0   1.0010  3.00       0.45      8.8        6
6.3                        0.30         0.34             1.6      0.049                 14.0                 132.0   0.9940  3.30       0.49      9.5        6
8.1                        0.28         0.40             6.9      0.050                 30.0                  97.0   0.9951  3.26       0.44     10.1        6
7.2                        0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
7.2                        0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
'''

# 결측치 없음.

'''
print(wine.isnull().sum())

volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64
'''
# numpy 처리
wine    = wine.values
print(type(wine))

# 데이터
x = wine[:,:9]
y = wine[:,10]
print(x.shape)   #(4898, 9)
print(y.shape)   #(4898, )

np.save('./data/x_wine.npy', arr=x)
np.save('./data/y_wine.npy', arr=y)
np.info(y)


from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



# 트레인 테스트 스플릿
x_train,x_test,y_train,y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

# 전처리
scaler = StandardScaler()
scaler.fit(x)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)



# 모델    
 
model = RandomForestClassifier()              #score =  0.710204081632653, acc =  0.710204081632653



# 훈련 
model.fit(x_train, y_train)
score = model.score(x_test, y_test) #evaluate 같은거
print('score = ', score)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)  
print("acc = ", acc)


