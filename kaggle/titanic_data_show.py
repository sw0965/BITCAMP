import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 로드
train_data = pd.read_csv('./data/kaggle_csv/train.csv', index_col = 0, header=0, sep=',', encoding='CP949')
test_data = pd.read_csv('./data/kaggle_csv/test.csv', index_col = 0, header=0, sep=',', encoding='CP949')

# print(train_data)
# print(test_data)
# print('train_data : ', train_data.shape)  #(891, 11)
# print('test_data : ' ,test_data.shape)    #(418, 10)

'''
----------train_data.info()----------

<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Survived  891 non-null    int64    생존
 1   Pclass    891 non-null    int64    티켓등급
 2   Name      891 non-null    object   이름
 3   Sex       891 non-null    object   성별
 4   Age       714 non-null    float64  나이      결측치 177
 5   SibSp     891 non-null    int64    형제
 6   Parch     891 non-null    int64    부모
 7   Ticket    891 non-null    object   티켓번호
 8   Fare      891 non-null    float64  지불한 운임
 9   Cabin     204 non-null    object   객실번호  결측치 887
 10  Embarked  889 non-null    object   어디서 탔는지    결측치 2
dtypes: float64(2), int64(4), object(5)


-----------test_data.info()-----------
<class 'pandas.core.frame.DataFrame'>
Int64Index: 418 entries, 892 to 1309
Data columns (total 10 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Pclass    418 non-null    int64
 1   Name      418 non-null    object
 2   Sex       418 non-null    object
 3   Age       332 non-null    float64
 4   SibSp     418 non-null    int64
 5   Parch     418 non-null    int64
 6   Ticket    418 non-null    object
 7   Fare      417 non-null    float64
 8   Cabin     91 non-null     object
 9   Embarked  418 non-null    object
dtypes: float64(2), int64(3), object(5)
'''

