import numpy as np
import matplotlib.pyplot as plt

def selu(x):
    return scale*(x>=0)*x + (x<0)*0.01*(np.exp(x)-1)

x = np.arange(-5,5,0.1)
y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# 파라미터도 존재한다. 
