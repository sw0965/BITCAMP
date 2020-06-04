import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns            # seaborn 그래프처럼 보여줄때 쓰는 것.
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('./data/kaggle_csv/train.csv')
test = pd.read_csv('./data/kaggle_csv/test.csv')


# Age 의 Nan 값
age_nan_rows = train[train['Age'].isnull()]  # Age에 Nan값을 가진 사람만 출력
'''
print(age_nan_rows.head(5))

    PassengerId  Survived  Pclass                           Name     Sex  Age  SibSp  Parch  Ticket     Fare Cabin Embarked
5             6         0       3               Moran, Mr. James    male  NaN      0      0  330877   8.4583   NaN        Q
17           18         1       2   Williams, Mr. Charles Eugene    male  NaN      0      0  244373  13.0000   NaN        S
19           20         1       3        Masselmani, Mrs. Fatima  female  NaN      0      0    2649   7.2250   NaN        C
26           27         0       3        Emir, Mr. Farred Chehab    male  NaN      0      0    2631   7.2250   NaN        C
28           29         1       3  O'Dwyer, Miss. Ellen "Nellie"  female  NaN      0      0  330959   7.8792   NaN        Q
'''
# Sex 값 0, 1로 분류                         *LabelEncoder : target labels with value between 0 and n_classes-1.  (n_classes -1?)
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
test['Sex']  = LabelEncoder().fit_transform(test['Sex'])

'''
print(train.head())
   PassengerId  Survived  Pclass                                               Name  Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    1  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina    0  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  35.0      1      0            113803  53.1000  C123        S
4            5         0       3                           Allen, Mr. William Henry    1  35.0      0      0            373450   8.0500   NaN        S

print(test.head())
   PassengerId  Pclass                                          Name  Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
0          892       3                              Kelly, Mr. James    1  34.5      0      0   330911   7.8292   NaN        Q
1          893       3              Wilkes, Mrs. James (Ellen Needs)    0  47.0      1      0   363272   7.0000   NaN        S
2          894       2                     Myles, Mr. Thomas Francis    1  62.0      0      0   240276   9.6875   NaN        Q
3          895       3                              Wirz, Mr. Albert    1  27.0      0      0   315154   8.6625   NaN        S
4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)    0  22.0      1      1  3101298  12.2875   NaN        S
'''

train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train['Name'].unique()
titles
test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test['Name'].unique()
test_titles

'''
print(titles)
['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
 'Sir' 'Mlle' 'Col' 'Capt' 'the Countess' 'Jonkheer']
 
print(test_titles)
['Mr' 'Mrs' 'Miss' 'Master' 'Ms' 'Col' 'Rev' 'Dr' 'Dona']
'''