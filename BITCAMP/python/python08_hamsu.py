# sum이란 함수를 만들겠다 그리고 a와 b 를 받아들이겠다.  def = 정의하겠다.

def sum1(a, b):
    return a+b     # 되돌려주겠다
a = 1
b = 2 
c = sum1(a, b)

print(c)


def sum1(c, d):
    return c+d     
a = 1
b = 2 
c = sum1(a, b)

print(c)

###곱셈, 나눗셈, 뺄셈 함수를 만드시오.
# mul1, div1, sub1
def mul1(a, b):
    return a*b
a = 2
b = 3
c = mul1(a, b)
print(c)


def div1(a, b):
    return a/b
a = 2
b = 4
c = div1(a, b)
print(c)


def sub1(a, b):
    return a-b
a = 3
b = 4
c = sub1(a, b)
print(c)

def sayYeh():
    return 'hi'

aaa = sayYeh()
print(aaa)

#def 뒤에 들어가는 a,b 매개변수

def sum2(a, b, c):
    return a + b + c

a = 1
b = 2
c = 34
d = sum2(a, b, c)

print(d)