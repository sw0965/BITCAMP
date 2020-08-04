# https://www.acmicpc.net/problem/19532
#----------수학은 비대면강의입니다  문제 (브3)-------------#



# 정답

a,b,c,d,e,f = map(int,input().split())

x = (c*e-b*f)/(a*e-b*d)
y = (a*f-c*d)/(a*e-b*d)
print(int(x),int(y))



# y = (c*d-f*a)//(b*d-e*a)
# x = (c - (b*y))//a

# print(x, y)
# x = c-by/a

# x = -999<= x <= 999
# fir = a*x + b*y == c
# sec = d*x + e*y == f
# print(x)
# print(y)

# if -999<=int(x)<=999 and -999<=int(y)<=999: 
#     if ax+by == c and dx+ey==f:
#         if (ax-dx)+(by-ey)==(c-f):
#             print(x, y)