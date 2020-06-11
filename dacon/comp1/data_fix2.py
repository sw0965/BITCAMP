import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

src = np.load('./data/dacon/bio/src.npy', allow_pickle='ture')
dst = np.load('./data/dacon/bio/dst.npy', allow_pickle='ture')


print(dst)  
print(dst.shape)  # 10000 35
print(src)
print(src.shape)  # 10000 35

# nan을 제거한뒤 dst는 값이 이상하고 ,src는 이상치가 존재한다

plt.plot(src)
plt.show()
