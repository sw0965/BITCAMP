gradient = lambda x: 2*x -4     # lambda 간략화된 함수 x는 인풋하는 함수 : 후 내가 쓰고싶은 함수 쓰면 됌

def gradient2(x):               # 위에 함수랑 같다. 
    temp = 2*x -4
    return temp

x = 3

print(gradient(x))

print(gradient2(x))