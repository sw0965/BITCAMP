# 2번정답
# x = [1, 2, 3]
# x = x - 1
# print(x)   그냥 파이썬으로는 오류뜸



import numpy as np  # 넘파이 사용하면 오류 안뜸
'''
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1  #인수를 뺌 그럼 01234 01234가 됌 
print(y)

from keras.utils import np_utils
y = np_utils.to_categorical(y)
print(y)
print(y.shape)
'''
# 2번의 두번째 답
y = np.array([1,2,3,4,5,1,2,3,4,5])   # (10, )   #y의 차원을 바꿔줘야함(와꾸를 맞춰줘야됌)
# y = y.reshape(-1, 1)
y = y.reshape(10, 1)
from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
print(y)
print(y.shape)