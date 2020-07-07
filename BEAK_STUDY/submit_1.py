# https://www.acmicpc.net/problem/13136
#-----------Do Not Touch Anything  문제 (브4)-------------#
# 쌩쇼
# x, y, cctv = map(int,input().split())
# a = x%cctv
# b = y%cctv
# c = cctv
# d = cctv
# if a != 0 and b != 0:
#     c = cctv + 1
#     d = cctv + 1
#     print(c*d)

# elif a == 0 and b != 0:
#     c = cctv
#     d = cctv+1
#     print(c*d)

# elif a != 0 and b == 0:
#     c = cctv+1
#     d = cctv
#     print(c*d)


# 정답
import math
x, y, cctv = map(int,input().split())

a = math.ceil(x/cctv)
b = math.ceil(y/cctv)

print(a*b)