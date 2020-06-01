import numpy as np
import pandas as pd

df1 = pd.read_csv('./data/kospi200.csv', index_col=0, header=0, encoding='cp949', sep=',')

print('df1 : ',df1)
print(df1.shape)

df2 = pd.read_csv('./data/samsung.csv', index_col=0, header=0, encoding='cp949', sep=',')

print(df2)
print(df2.shape)


#코스피의 거래량
for i in range(len(df1.index)):
    df1.iloc[i,4] = int(df1.iloc[i,4].replace(',',''))

#삼선전자의 모든 데이터
for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):
        df2.iloc[i,j] = int(df2.iloc[i,j].replace(',',''))

# 오름차순으로 바꾸기
df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

# numpy 변환
df1 = df1.values
df2 = df2.values

print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./model/data/kospi200.npy', arr=df1)
np.save('./model/data/samsung.npy', arr=df2)



