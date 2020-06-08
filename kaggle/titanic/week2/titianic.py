import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns            # seaborn 그래프처럼 보여줄때 쓰는 것.
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('./data/kaggle_csv/titanic/train.csv')
test = pd.read_csv('./data/kaggle_csv/titanic/test.csv')

'''
NAN의 갯수

print('show1 :', train.isnull().sum())
show1 : PassengerId      0
Survived         0        
Pclass           0        
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

print('show2 :',test.isnull().sum())
show2 : PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
'''
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
titles        # train의 이름분리
test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_titles = test['Name'].unique()
test_titles   # test의 이름분리

'''
print(titles)
['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
 'Sir' 'Mlle' 'Col' 'Capt' 'the Countess' 'Jonkheer']
 
print(test_titles)
['Mr' 'Mrs' 'Miss' 'Master' 'Ms' 'Col' 'Rev' 'Dr' 'Dona']
'''

# Age 결측치 채워주기위해 아까 분리해둔 이름으로 중간값 구하기
train['Age'].fillna(-1, inplace=True)  # nan age값을 -1로
test['Age'].fillna(-1, inplace=True)
# print(train['Age'])
print(test['Age'])



medians = dict()
for title in titles:
    median = train.Age[(train["Age"] != -1) & (train['Name'] == title)].median()  # train의 age가 -1 그리고 train의 name이 타이틀과 같으면
    medians[title] = median

print(medians[title])  # 평균 38.0

# train, test 의 만약 age 행이 -1과 같으면 train, test Age 값과 Name값 평균이랑 같다?
for index, row in train.iterrows():
    if row['Age'] == -1:
        train.loc[index, 'Age'] = medians[row['Name']]  # loc 컬럼 네임으로  (iloc = 리스트 숫자)

for index, row in test.iterrows():
    if row['Age'] == -1:
        test.loc[index, 'Age'] = medians[row['Name']]

'''
print(medians)
  name의 평균값을 왜 구하는지 모르겠다. (나이가 -1(결측치일때) 나이랑 평균이름(나눈이름)이랑 같다)
  결측값이 아닌 나이와 이름을 평균으로 묶어서 예상값을 다시 나이값에 치환해주기 위해 사용?

{'Mr': 30.0, 'Mrs': 35.0, 'Miss': 21.0, 'Master': 3.5, 'Don': 40.0, 'Rev': 46.5, 'Dr': 46.5, 'Mme': 24.0, 'Ms': 28.0, 'Major': 48.5, 'Lady': 48.0, 'Sir': 49.0, 'Mlle': 24.0, 
'Col': 58.0, 'Capt': 70.0, 'the Countess': 33.0, 'Jonkheer': 38.0}

print(train.isnull().sum())
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64


print(test.isnull().sum())
PassengerId      0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64

'''
test_age_nan_rows = test[test['Age'].isnull()]

print(test_age_nan_rows.head(5))

print(train.head())
'''
   PassengerId  Survived  Pclass  Name  Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3    Mr    1  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1   Mrs    0  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3  Miss    0  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1   Mrs    0  35.0      1      0            113803  53.1000  C123        S
4            5         0       3    Mr    1  35.0      0      0            373450   8.0500   NaN        S
'''

fig = plt.figure(figsize=(15,6))

i=1
for title in train['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')
    i += 1
# plt.show()

# 각 이름별로 많이 죽은성에서 적게 죽은 성 순서로
title_replace = {
    'Don':0,
    'Rev':0,
    'Capt':0,
    'Jonkheer':0,
    'Mr':1,
    'Dr':2,
    'Major':3,
    'Col':3,
    'Master':4,
    'Miss':5,
    'Mrs':6,
    'Mme':7,
    'Ms':7,
    'Lady':7,
    'Sir':7,
    'Mlle':7,
    'the Countess':7
}


'''
print(test['Name'].unique())

train--
['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
 'Sir' 'Mlle' 'Col' 'Capt' 'the Countess' 'Jonkheer']

test--  train에는 Dona가 없다.
 ['Mr' 'Mrs' 'Miss' 'Master' 'Ms' 'Col' 'Rev' 'Dr' 'Dona']
 '''

'''
print(test[test['Name'] == 'Dona']) # test name의 도나를 출력

      PassengerId  Pclass  Name  Sex   Age  SibSp  Parch    Ticket   Fare Cabin Embarked
414         1306       1  Dona    0    39.0     0     0   PC 17758  108.9  C105        C
'''


train['Name'] = train['Name'].apply(lambda x: title_replace.get(x)) # 람다함수 : 식을 간결하게 만들어주는거?
test['Name'] = test['Name'].apply(lambda x: title_replace.get(x))




'''   # test 네임칸에 결측치가 생김
print(test.isnull().sum())
PassengerId      0
Pclass           0
Name             1
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
'''
'''
print(test[test['Name'].isnull()])
     PassengerId  Pclass  Name  Sex   Age  SibSp  Parch    Ticket   Fare Cabin Embarked
414         1306       1   NaN    0  39.0      0      0  PC 17758  108.9  C105        C
'''

# Dona 성을 가진 사람 해당 성별에 name에 대한 중간값을 넣어주기

print(test[test['Sex'] == 0]['Name'].mean())
#5.490066225165563


print(train[train['Sex'] == 0]['Name'].mean())
#5.426751592356688

print(test[test['Name'].isnull()]['Sex'])
# 414    0
#Name: Sex, dtype: int32
print(test[test['Name'].isnull()]['Name'])
#414   NaN
# Name: Name, dtype: float64
test['Name'] = test['Name'].fillna(value=train[train['Sex'] == 0]['Name'].mean())
'''
print(test.head())

   PassengerId  Pclass  Name  Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
0          892       3   1.0    1  34.5      0      0   330911   7.8292   NaN        Q
1          893       3   6.0    0  47.0      1      0   363272   7.0000   NaN        S
2          894       2   1.0    1  62.0      0      0   240276   9.6875   NaN        Q
3          895       3   1.0    1  27.0      0      0   315154   8.6625   NaN        S
4          896       3   6.0    0  22.0      1      1  3101298  12.2875   NaN        S
'''

print(test.isnull().sum())
'''
PassengerId      0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
'''

print(train.isnull().sum())
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''

train_test_data = [train, test]

# 나이 세분화
for dataset in train_test_data:
    dataset.loc[ dataset['Age']<=10, 'Age'] = 0,
    dataset.loc[(dataset['Age']>10)&(dataset['Age']<=16), 'Age'] = 1,
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=20), 'Age'] = 2,
    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=26), 'Age'] = 3,
    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=30), 'Age'] = 4,
    dataset.loc[(dataset['Age']>30)&(dataset['Age']<=36), 'Age'] = 5,
    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=40), 'Age'] = 6,
    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=46), 'Age'] = 7,
    dataset.loc[(dataset['Age']>46)&(dataset['Age']<=50), 'Age'] = 8,
    dataset.loc[(dataset['Age']>50)&(dataset['Age']<=60), 'Age'] = 9,
    dataset.loc[ dataset['Age']>60, 'Age'] = 10

# 나이별 죽은사람과 산사람 이미지 확인
fig = plt.figure(figsize=(15,6))

i=1
for age in train['Age'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Age : {}'.format(age))
    train.Survived[train['Age'] == age].value_counts().plot(kind='pie')
    i += 1
# plt.show()

age_point_replace = {
    0: 8,
    1: 6,
    2: 2,
    3: 4,
    4: 1,
    5: 7,
    6: 3,
    7: 2,
    8: 5,
    9: 4,
    10: 0
}

for dataset in train_test_data:
    dataset['age_point'] = dataset['Age'].apply(lambda x: age_point_replace.get(x))

'''
print(train.head())

   PassengerId  Survived  Pclass  Name  Sex  Age  SibSp  Parch            Ticket     Fare Cabin Embarked  age_point
0            1         0       3     1    1  3.0      1      0         A/5 21171   7.2500   NaN        S          4
1            2         1       1     6    0  6.0      1      0          PC 17599  71.2833   C85        C          3
2            3         1       3     5    0  3.0      0      0  STON/O2. 3101282   7.9250   NaN        S          4
3            4         1       1     6    0  5.0      1      0            113803  53.1000  C123        S          7
4            5         0       3     1    1  5.0      0      0            373450   8.0500   NaN        S          7


print(test.head())
   PassengerId  Pclass  Name  Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked  age_point
0          892       3   1.0    1   5.0      0      0   330911   7.8292   NaN        Q          7
1          893       3   6.0    0   8.0      1      0   363272   7.0000   NaN        S          5
2          894       2   1.0    1  10.0      0      0   240276   9.6875   NaN        Q          0
3          895       3   1.0    1   4.0      0      0   315154   8.6625   NaN        S          1
4          896       3   6.0    0   3.0      1      1  3101298  12.2875   NaN        S          4
'''
#Embarked가 nan인 사람은 S로 채워주자. 이를 보다 근거있는 값으로 채울 수는 없는지 고민해보자.
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

#Sibp 와 Parch값을 이용하여 Family size 추가
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Familysize가 4보다 큰 남자는 아버지일 확률으로 잡고
maybe_dad_mask = (train['FamilySize'] > 4) & (train['Sex'] == 1)
'''
print(maybe_dad_mask.head())

0    False
1    False
2    False
3    False
4    False
dtype: bool
'''
train['maybe_dad'] = 1

train.loc[maybe_dad_mask,'maybe_dad'] = 0
# 아버지로 분류된 사람과 아닌 사람의 생존 비율
fig = plt.figure()
ax1 = train.Survived[train['maybe_dad'] == 1].value_counts().plot(kind='pie')
ax2 = train.Survived[train['maybe_dad'] == 0].value_counts().plot(kind='pie')

test['maybe_dad'] = 1
test_maybe_dad_mask = (test['FamilySize'] > 4) & (test['Sex'] == 1)
test.loc[test_maybe_dad_mask,'maybe_dad'] = 0

#Family size에 새로운 값 입히기. 그럴려면 생존비율 확인
train['FamilySize'].unique()

test['FamilySize'].unique()

fig = plt.figure(figsize=(15,6))

i=1
for size in train['FamilySize'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Size : {}'.format(size))
    train.Survived[train['FamilySize'] == size].value_counts().plot(kind='pie')
    i += 1

size_replace = {
    1: 3,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 1,
    7: 4,
    8: 0,
    11: 0
}

for dataset in train_test_data:
    dataset['fs_point'] = dataset['FamilySize'].apply(lambda x: size_replace.get(x))
    dataset.drop('FamilySize',axis=1,inplace=True)

'''
print(train.isnull().sum())
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         0
age_point        0
maybe_dad        0
fs_point         0
dtype: int64


print(test.isnull().sum())
PassengerId      0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
age_point        0
maybe_dad        0
fs_point         0
dtype: int64
'''
#Pclass 별로 생존비율 확인하고 값 넣기
fig = plt.figure(figsize=(15,6))

i=1
for x in train['Pclass'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Pclass : {}'.format(x))
    train.Survived[train['Pclass'] == x].value_counts().plot(kind='pie')
    i += 1
# plt.show()

for dataset in train_test_data:
    dataset.loc[dataset['Pclass']==3,'Pclass_point'] = 0
    dataset.loc[dataset['Pclass']==2,'Pclass_point'] = 1
    dataset.loc[dataset['Pclass']==1,'Pclass_point'] = 2

fig = plt.figure(figsize=(15,6))

# Embarked
i=1
for x in train['Embarked'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Em : {}'.format(x))
    train.Survived[train['Embarked'] == x].value_counts().plot(kind='pie')
    i += 1
# plt.show()

for dataset in train_test_data:
    dataset.loc[dataset['Embarked']==0,'Em_point'] = 0
    dataset.loc[dataset['Embarked']==2,'Em_point'] = 1
    dataset.loc[dataset['Embarked']==1,'Em_point'] = 2

train['Cabin'].unique()
'''
print(train['Cabin'].unique())
[nan 'C85' 'C123' 'E46' 'G6' 'C103' 'D56' 'A6' 'C23 C25 C27' 'B78' 'D33'
 'B30' 'C52' 'B28' 'C83' 'F33' 'F G73' 'E31' 'A5' 'D10 D12' 'D26' 'C110'
 'B58 B60' 'E101' 'F E69' 'D47' 'B86' 'F2' 'C2' 'E33' 'B19' 'A7' 'C49'
 'F4' 'A32' 'B4' 'B80' 'A31' 'D36' 'D15' 'C93' 'C78' 'D35' 'C87' 'B77'
 'E67' 'B94' 'C125' 'C99' 'C118' 'D7' 'A19' 'B49' 'D' 'C22 C26' 'C106'
 'C65' 'E36' 'C54' 'B57 B59 B63 B66' 'C7' 'E34' 'C32' 'B18' 'C124' 'C91'
 'E40' 'T' 'C128' 'D37' 'B35' 'E50' 'C82' 'B96 B98' 'E10' 'E44' 'A34'
 'C104' 'C111' 'C92' 'E38' 'D21' 'E12' 'E63' 'A14' 'B37' 'C30' 'D20' 'B79'
 'E25' 'D46' 'B73' 'C95' 'B38' 'B39' 'B22' 'C86' 'C70' 'A16' 'C101' 'C68'
 'A10' 'E68' 'B41' 'A20' 'D19' 'D50' 'D9' 'A23' 'B50' 'A26' 'D48' 'E58'
 'C126' 'B71' 'B51 B53 B55' 'D49' 'B5' 'B20' 'F G63' 'C62 C64' 'E24' 'C90'
 'C45' 'E8' 'B101' 'D45' 'C46' 'D30' 'E121' 'D11' 'E77' 'F38' 'B3' 'D6'
 'B82 B84' 'D17' 'A36' 'B102' 'B69' 'E49' 'C47' 'D28' 'E17' 'A24' 'C50'
 'B42' 'C148']
 '''
#cabin이 nan값인 사람들 u로 채우기 (u = nan을 나타내는 문자열)
#fare 데이터와 비교 fare의 nan 값은 1개라서 일단 0으로 기입.
for data in train_test_data:
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    data['Cabin'].unique()
    data['Fare'].fillna(0,inplace=True)
    data['Fare'] = data['Fare'].apply(lambda x: int(x))


#cabin 별로 생존비율 확인
fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1
# plt.show()

#fare 값도 다양해서 구간별로 나누어 보기
temp = train['Fare'].unique()
temp.sort()
temp
'''
print(temp)
[  0   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
  39  40  41  42  46  47  49  50  51  52  53  55  56  57  59  61  63  65
  66  69  71  73  75  76  77  78  79  80  81  82  83  86  89  90  91  93
 106 108 110 113 120 133 134 135 146 151 153 164 211 221 227 247 262 263
 512]
'''
# fare 분류
for dataset in train_test_data:
    dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0,
    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=80), 'Fare'] = 1,
    dataset.loc[(dataset['Fare']>80)&(dataset['Fare']<=100), 'Fare'] = 2,
    dataset.loc[(dataset['Fare']>100), 'Fare'] = 3

# 아까 cabin의 값을 u로 넣어준 사람들에게 존재하는 cabin값으로 넣어주기 위해 범위가 많은 곳을 해당 값으로 u값을 대체
fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1
# plt.show()

for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 0), 'Cabin'] = 'G',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 1), 'Cabin'] = 'T',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 2), 'Cabin'] = 'C',
    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 3), 'Cabin'] = 'B',


fig = plt.figure(figsize=(15,6))

i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1

fig = plt.figure(figsize=(15,6))

# cabin값 별로 생존여부 확인
i=1
for x in train['Cabin'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Cabin : {}'.format(x))
    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')
    i += 1
   #  plt.show()


for dataset in train_test_data:
    dataset.loc[(dataset['Cabin'] == 'G'), 'Cabin_point'] = 0,
    dataset.loc[(dataset['Cabin'] == 'C'), 'Cabin_point'] = 3,
    dataset.loc[(dataset['Cabin'] == 'E'), 'Cabin_point'] = 5,
    dataset.loc[(dataset['Cabin'] == 'T'), 'Cabin_point'] = 1,
    dataset.loc[(dataset['Cabin'] == 'D'), 'Cabin_point'] = 7,
    dataset.loc[(dataset['Cabin'] == 'A'), 'Cabin_point'] = 2,
    dataset.loc[(dataset['Cabin'] == 'B'), 'Cabin_point'] = 6,
    dataset.loc[(dataset['Cabin'] == 'F'), 'Cabin_point'] = 4,

#Fare 별로 생존여부확인 그리고 점수로 대체하기
fig = plt.figure(figsize=(15,6))

i=1
for x in train['Fare'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Fare : {}'.format(x))
    train.Survived[train['Fare'] == x].value_counts().plot(kind='pie')
    i += 1
   #  plt.show()

for dataset in train_test_data:
    dataset.loc[(dataset['Fare'] == 0), 'Fare_point'] = 0,
    dataset.loc[(dataset['Fare'] == 1), 'Fare_point'] = 1,
    dataset.loc[(dataset['Fare'] == 2), 'Fare_point'] = 3,
    dataset.loc[(dataset['Fare'] == 3), 'Fare_point'] = 2,




############################################# 정규화 #############################################
from sklearn.preprocessing import StandardScaler
for dataset in train_test_data:
    dataset['Name'] = StandardScaler().fit_transform(dataset['Name'].values.reshape(-1, 1))
    dataset['Sex'] = StandardScaler().fit_transform(dataset['Sex'].values.reshape(-1, 1))
    dataset['maybe_dad'] = StandardScaler().fit_transform(dataset['maybe_dad'].values.reshape(-1, 1))
    dataset['fs_point'] = StandardScaler().fit_transform(dataset['fs_point'].values.reshape(-1, 1))
    dataset['Em_point'] = StandardScaler().fit_transform(dataset['Em_point'].values.reshape(-1, 1))
    dataset['Cabin_point'] = StandardScaler().fit_transform(dataset['Cabin_point'].values.reshape(-1, 1))
    dataset['Pclass_point'] = StandardScaler().fit_transform(dataset['Pclass_point'].values.reshape(-1, 1))
    dataset['age_point'] = StandardScaler().fit_transform(dataset['age_point'].values.reshape(-1, 1))
    dataset['Fare_point'] = StandardScaler().fit_transform(dataset['Fare_point'].values.reshape(-1, 1))

# 필요 없는거 없애기
train.drop(['PassengerId','Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)
test.drop(['Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)

'''
print(train.head())
   Survived      Name       Sex  age_point  maybe_dad  fs_point  Pclass_point  Em_point  Cabin_point  Fare_point
0         0 -0.797294  0.737695   0.122488   0.183419  0.894514     -0.827377 -0.585954    -0.562920   -0.512784
1         1  1.537975 -1.355574  -0.294065   0.183419  0.894514      1.566107  1.942303     0.840704    0.914998
2         1  1.070922 -1.355574   0.122488   0.183419 -0.523657     -0.827377 -0.585954    -0.562920   -0.512784
3         1  1.537975 -1.355574   1.372147   0.183419  0.894514      1.566107 -0.585954     0.840704    0.914998
4         0 -0.797294  0.737695   1.372147   0.183419 -0.523657     -0.827377 -0.585954    -0.562920   -0.512784
'''

train_data = train.drop('Survived', axis=1)
target = train['Survived']

'''
print(train_data.head())
       Name       Sex  age_point  maybe_dad  fs_point  Pclass_point  Em_point  Cabin_point  Fare_point
0 -0.797294  0.737695   0.122488   0.183419  0.894514     -0.827377 -0.585954    -0.562920   -0.512784
1  1.537975 -1.355574  -0.294065   0.183419  0.894514      1.566107  1.942303     0.840704    0.914998
2  1.070922 -1.355574   0.122488   0.183419 -0.523657     -0.827377 -0.585954    -0.562920   -0.512784
3  1.537975 -1.355574   1.372147   0.183419  0.894514      1.566107 -0.585954     0.840704    0.914998
4 -0.797294  0.737695   1.372147   0.183419 -0.523657     -0.827377 -0.585954    -0.562920   -0.512784


print(target.head())
0    0
1    1
2    1
3    1
4    0
Name: Survived, dtype: int64
'''

print(train.shape)      #(981, 10)
print(test.shape)       #(418, 10)
print(train_data.shape) #(891, 9)
x = train
y = test
x_predict = train_data
############################################# 모델링 #############################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.2)

# scaler = StandardScaler()
# scaler.fit(x_train)   
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_predict = scaler.transform(x_predict)

#. 트레인 테스트 분류
parameters = {'n_estimators' : [10], 'max_depth' : [1, 10, 100, 1000],'max_features':['auto'],
'max_leaf_nodes':[None], 'class_weight':[None], 'criterion':['gini'],
'min_impurity_decrease':[0.0], 'min_impurity_split' : [None],
'min_samples_leaf':[1], 'min_samples_split':[2], 'warm_start':[False],
'min_weight_fraction_leaf': [0.0],'bootstrap':[True], 'n_jobs':[None],
'oob_score':[False], 'random_state':[None], 'verbose':[0]}
    

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1)# cv= 5라고 써도됌 

model.fit(x, y)

print("최적의 매개변수 : ", model.best_estimator_)

''' 
#내 파라미터중 제일 좋은거 찾았을때 밑에 나온게 결괏값
'''


y_pred = model.predict(x_predict)
print("최종 정답률 : ", accuracy_score(y, y_pred)) # 뭐가 acc : 1이 나온지 모름