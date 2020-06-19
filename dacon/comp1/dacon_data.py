#데이컨 전처리 방법
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import missingno as msno

############### 데이터 불러오기 ###############
train      = pd.read_csv('./DATA/dacon/bio/train.csv',index_col=0, header=0)
test  = pd.read_csv('./DATA/dacon/bio/test.csv', index_col=0, header=0)
y_predict  = pd.read_csv('./DATA/dacon/bio/sample_submission.csv', index_col=0, header=0)


print(train.shape)
# print(train.head())

# print(train.isnull().sum()[train.isnull().sum().values > 0])

# test.filter(regex='_src$',axis=1).head().T.plot()
# # plt.show()

# train_dst = train.filter(regex='_dst$', axis=1).replace(0, np.NaN) # dst 데이터만 따로 뺀다.
# test_dst = test.filter(regex='_dst$', axis=1).replace(0, np.NaN) # 보간을 하기위해 결측값을 삭제한다.
# print(test_dst.head())