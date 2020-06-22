# src, dst 전체 데이터 dst만 처리중.
#데이컨 전처리 방법


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import missingno as msno

############### 데이터 불러오기 ###############
train      = pd.read_csv('./DATA/dacon/bio/train.csv',index_col=0, header=0)
test  = pd.read_csv('./DATA/dacon/bio/test.csv', index_col=0, header=0)
y_predict  = pd.read_csv('./DATA/dacon/bio/sample_submission.csv', index_col=0, header=0)


print(train.shape)     #(10000, 75)
print(test.shape)      #(10000, 71)
print(y_predict.shape) #(10000, 4)



print(train.isnull().sum()[train.isnull().sum().values > 0])
print(test.isnull().sum()[test.isnull().sum().values > 0])


print(train.filter(regex='_dst$').head())
print('______________________________________________________')

train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN) # dst 데이터만 따로 뺀다.
test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
print(train_dst.head())
print('______________________________________________________')
train_dst = train_dst.interpolate(methods = 'linear', axis= 1)
test_dst = test_dst.interpolate(methods = 'linear', axis= 1)
print(train_dst.head())

print('______________________________________________________')
train_dst.fillna(0, inplace=True)
test_dst.fillna(0, inplace=True)

train.update(train_dst)
test.update(test_dst)

train.to_csv('./DACON/comp1/data/train_fix.csv')
test.to_csv('./DACON/comp1/data/test_fix.csv')

np.save('./DACON/comp1/data/train.npy', arr=train)
np.save('./DACON/comp1/data/test.npy', arr=test)
np.save('./DACON/comp1/data/y_predict.npy', arr=y_predict)