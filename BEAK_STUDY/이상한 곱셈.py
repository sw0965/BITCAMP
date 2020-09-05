# https://www.acmicpc.net/problem/1225
#-----------  이상한 곱셈  문제 (브2)-------------#

# a, b= input().split()

# a_ls, b_ls = [], []
# c, d = 0
# for i, j in len(a), len(b):
#     a_ls.append(a[0])
#     c =+ 1
#     b_ls.append(b[0])
#     d =+ 1

#     print(a_ls)
#     print(b_ls)



# a = '1234'
# print(len(a))
# print(list(a))
# for i in list(a):
#     print(a)

# a, b = input().split()

# a,b = list(a), list(b)
# print(a, b)
# a = int(a)
# print(a)

# for i in int(a):
#     print(i)

# a, b = list(map(int().split()))
# print(a, b)

# time out
# a, b = input().split()

# # a,b = list(map(int, a)), list(map(int, b))
# # print(a, b)
# # a[0]*b[0]
# # a[0]*b[1]
# results = []
# for i in list(map(int,a)):
#     for j in list(map(int, b)):
#         # print(sum(results.append(i*j)))
#         result = i*j
#         results.append(result)
# print(sum(results))
#         # sum(results)

'''# time out'''
'''import sys
a, b = sys.stdin.readline().split()


results = []
for i in list(map(int,a)):
    for j in list(map(int, b)):
        
        result = i*j
        results.append(result)

print(sum(results))'''

# 규칙을 찾아보기로 함
'''정답'''
import sys

a, b = sys.stdin.readline().split()

a_ls = []
for i in list(map(int,a)):
    a_ls.append(i)
    c = sum(a_ls)
b_ls = []
for j in list(map(int,b)):
    b_ls.append(j)
    d = sum(b_ls)
print(c*d)
# print(sum(ls))
    # sum(ls)
    # print(c)
    # d = ls.append(j)

#     print(i)
#     sum_a = sum(i)
# print(sum_a)