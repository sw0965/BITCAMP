import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()


"""Arguments

x: 입력 텐서.
alpha: 부동소수점. 음수 부분의 기울기. 디폴트는 0
max_value: 부동소수점. 포화 임계값
threshold: 부동소수점. 임계값 활성화를 위한 임계치"""