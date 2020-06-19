import numpy as np
import pandas as pd

load_hite = pd.read_csv('./data/csv/Hite.csv', index_col=0, header=0, encoding='cp949', sep=',')
print(type(load_hite))
'''
drop_hite = load_hite.dropna(how='all')
print(drop_hite)


'''