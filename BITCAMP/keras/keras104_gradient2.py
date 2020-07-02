import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100)
y = f(x)

# 그림
plt.plot(x, y, 'k--')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

gradient = lambda x : 2*x - 4

x0 = 0.0
MaxIter = 10
learning_rate = 0.25

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))
'''
step    x       f(x)
00      0.00000 6.00000
f = lambda x : x**2 - 4*x + 6   # 스텝 = x가 0일때 
'''

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))
    '''
    경사 하강법 구하는 생코딩

    step    x       f(x)
    00      0.00000 6.00000
    01      1.00000 3.00000
    02      1.50000 2.25000
    03      1.75000 2.06250
    04      1.87500 2.01562
    05      1.93750 2.00391
    06      1.96875 2.00098
    07      1.98438 2.00024
    08      1.99219 2.00006
    09      1.99609 2.00002
    10      1.99805 2.00000
    '''