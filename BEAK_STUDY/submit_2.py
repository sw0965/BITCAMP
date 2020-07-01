# https://www.acmicpc.net/problem/2742
#------------- 기찍 N  문제 (브3)-------------#
import sys
N = int(sys.stdin.readline())
for i in list(range(1,N+1)[::-1]):
    print(i)
'''
# [::-1] keyworld 차순을 바꿔준다.
# print(list(range(1, 5)[::-1]))  # [4, 3, 2, 1]
'''