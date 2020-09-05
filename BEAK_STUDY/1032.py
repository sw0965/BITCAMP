# import sys

# # a = ['config.sys', 'config.inf', 'configures']
# # b = {}
# # for i in a:
# #     try: b[i] += 1
# #     except: b[i] = 1
# # print(b)



# # c = int(input())

# # d = sys.stdin.readline()
# # e = sys.stdin.readline()
# # f = sys.stdin.readline()
# d = input()
# e = input()
# f = input()

# print(len(d))
# g= list(d)+list(e)+list(f)
# # print(g[0])

import sys

N = int(sys.stdin.readline())
namelist = list()
for i in range(N):
    name = input()
    namelist.append(i)
print(namelist)