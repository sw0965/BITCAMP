import numpy as np
import pandas as pd

samsungE = np.load('./data/samE.npy', allow_pickle=True)
hite     = np.load('./data/hite.npy', allow_pickle=True)

print('samsung_data :', samsungE)
print('hite_dat :', hite)

print('samsung.shape : ', samsungE.shape)
print('hite.shape :', hite.shape)