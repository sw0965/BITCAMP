import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/wine/winequality-white.csv', index_col = 0, header=0, sep=';', encoding='CP949')

count_data = wine.groupby('quality')['quality'].count()  # 카운트하는 문법

print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''

count_data.plot()
plt.show()

# 5,6 이 몰려있다 축소를 시켜야됌