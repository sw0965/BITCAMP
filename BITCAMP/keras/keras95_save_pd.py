import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/iris.csv", index_col=None, header=0, sep=',')

print(datasets)

print(datasets.head())
print(datasets.tail())

print(datasets.values)   # 판다스를 넘파이로 바꾸는거 .values



aaa = datasets.values
print(type(aaa))

# x = aaa.data    #<class 'numpy.ndarray'>
# y = aaa.target  #<class 'numpy.ndarray'>
# print(x)
# print(y)


np.save('./data/iris_data.npy', arr=aaa)
# np.save('./data/iris_y.npy', arr=y)