# # https://www.acmicpc.net/problem/10797
# #------------10부제  문제 (브4)-------------#
# day = int(input())
# a,b,c,d,e = map(int,input().split())

# f = [a,b,c,d,e]
# c = []

# for i in f:
#     if i == day:
#         c.append(i)
# print(c.count(day))

# 7_03_main.py 
day = int(input())
count = 0

aaa = map(int,input().split())

for i in aaa:
    if i == day:
        count += 1
print(count)
