import numpy as np
import pandas as pd


df1 = pd.read_csv('./data/csv/SamsungE.csv', index_col=0, header=0, encoding='cp949', sep=',')#,keep_default_na=False)
df1 = df1.dropna(how='all')
print('samsung : ',df1)
print('df1 : ',df1.shape)  #(700, 1)


df2 = pd.read_csv('./data/csv/Hite.csv', index_col=0, header=0, encoding='cp949', sep=',')
df2 = df2.dropna(how='all')

df2 = df2.fillna('51,000')
print('hite : ',df2)
print('df2 : ',df2.shape)  #(720, 5)



for i in range(len(df1.index)):
    df1.iloc[i,0] = int(df1.iloc[i,0].replace(',',''))
print(df1)


for i in range(len(df2.index)):
    for j in range(len(df2.iloc[i])):
        df2.iloc[i,j] = int(df2.iloc[i,j].replace(',',''))
print(df2)


df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

df1 = df1.values
df2 = df2.values
print(type(df1), type(df2))  #509, 1
print(df1.shape, df2.shape)  #509, 5

np.save('./data/samE.npy', arr = df1)
np.save('./data/hite.npy', arr = df2)
