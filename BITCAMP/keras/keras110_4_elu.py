import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# 파라미터도 존재한다. 
"""
Arguments

x: 입력 텐서.
alpha: 스칼라, 음수 부분의 기울기"""


# def elu(x):
#     x = np.copy(x)
#     x[x<0]=0.2*(np.exp(x[x<0])-1)
#     return x


# a = 0.2
# x = np.arange(-5,5,0.1)
# y = [x if x>0 else a*(np.exp(x)-1) for x in x]


# def elu(x):
#     y_list = []
#     for x in x:
#         if(x>0):
#             y = x
#         if(x<0):
#             y = 0.2*(np.exp(x)-1)
#         y_list.append(y)
#     return y_list


# import numpy as np
# def elu(x):
#     if(x>0):
#         return x
#     if(x<0):
#         return 0.2*(np.exp(-x)-1)