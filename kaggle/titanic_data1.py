import pandas as pd
import numpy as np

train = pd.read_csv('./data/kaggle_csv/train.csv', index_col = 0, header=0, sep=',', encoding='CP949')
test = pd.read_csv('./data/kaggle_csv/test.csv', index_col = 0, header=0, sep=',', encoding='CP949')
# print(train)
# print('train data shape: ', train.shape)  #(891, 11)
# print('test data shape: ', test.shape)    #(418, 10)
# print('----------[train infomation]----------')
# print(train.info())
# print('----------[test infomation]----------')
# print(test.info())

# train.head(5)

train_and_test = [train, test]
for dataset in train_and_test:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.')

train['Title'].value()