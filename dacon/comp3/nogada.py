import numpy as np
import pandas as pd

# 주어진 데이터
x = pd.read_csv('./data/dacon/KAERI/train_features.csv', header=0, index_col=0)
y = pd.read_csv('./data/dacon/KAERI/train_target.csv', header=0, index_col=0)
#적용 데이터
x_pred = pd.read_csv('./data/dacon/KAERI/test_features.csv', header=0, index_col=0)
test_target = pd.read_csv('./data/dacon/KAERI/sample_submission.csv', header=0, index_col=0)

x = x.values        #array
y = y.values        #array
x_pred=x_pred.values

x = x[]