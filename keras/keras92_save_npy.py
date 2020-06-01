from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))     #<class 'sklearn.utils.Bunch'>

x_data = iris.data    #<class 'numpy.ndarray'>
y_data = iris.target  #<class 'numpy.ndarray'>

print(type(x_data))
print(type(y_data))

# 데이터 세이브
np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)

x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load))
print(type(y_data_load))
print(x_data_load.shape)
print(y_data_load.shape)


