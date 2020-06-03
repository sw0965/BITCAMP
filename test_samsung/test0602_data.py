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
print(samsung.shape)
hite = hite.fillna(method='bfill')  #back fill 전날값으로 채우겠다
hite = hite.dropna(axis=0)

#NaN 제거 2
# hite = hite[0:509]
# hite.iloc[0, 1:5] = [10,20,30,40]  # 인덱스 로케이션 0행에 1부터 5컬럼까지를 10 20 30 40만들어주겠다.
# hite.loc["2020-06-02", '고가':'거래량'] = ['10','20','30','40']
print(hite)

print(hite.shape)

# 삼성과 하이트의 정렬을 오름차순으로 
samsung = samsung.sort_values(['일자'], ascending=['True'])
hite    = hite.sort_values(['일자'], ascending=['True'])

print(samsung)
print(hite)

#콤마제거, 문자를 정수로 형변환

for i in range(len(samsung.index)):
    samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',',''))


for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',',''))

print(samsung.shape)  #(509, 1)
print(hite.shape)     #(509, 5)

samsung = samsung.values
hite    = hite.values

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)
# 이따가 ,랑 다른것도 넣어서 replace 해보기
