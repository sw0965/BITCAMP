import numpy as np 
import pandas as pd 

samsung = pd.read_csv('./data/csv/samsung.csv', index_col = 0, header=0, sep=',', encoding='CP949')

hite = pd.read_csv('./data/csv/hite.csv', index_col = 0, header=0, sep=',', encoding='CP949')

print(samsung.head())
print(hite.head())
print(samsung.shape)  #(700, 1)
print(hite.shape)     #(720, 5)

#NaN 제거 1
samsung = samsung.dropna(axis=0)
print(samsung)
print(samsung.shape)  #(509, 1)
# hite = hite.fillna(method='bfill')  #back fill 전날값으로 채우겠다
# hite = hite.dropna(axis=0)
# print(hite)
# print(hite.shape)    #(509, 5)

# str 변경
# for i in range(len(samsung.index)):
#     samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))
# print(samsung)


# for i in range(len(hite.index)):
#     for j in range(len(hite.iloc[i])):
#         hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))
# print(hite)

print('hite :',hite)
hite_x  = hite[0:]
print('hite_x :', hite_x)