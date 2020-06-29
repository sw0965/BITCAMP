import numpy as np 
import pandas as pd

data = pd.read_csv('./DACON/comp_JEJU/data/201901-202003.csv', header=0 )

# print(data.iloc[:,3])

Sectors = data.iloc[:,3]
print(Sectors)
# for i in Sectors:
#     if  == True:
#         print(Sectors.count())

for i in Sectors(x, y):
    if y == '건강보조식품 소매업':
        print(i)