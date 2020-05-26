import numpy as np

y_class = np.array([0, 1, 0, 1, 2]).reshape(-1, 1)
print(y_class)
print(y_class.shape)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_class)
