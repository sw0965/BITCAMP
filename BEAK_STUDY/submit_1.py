# https://www.acmicpc.net/problem/10250
#-----------  ACM 호텔  문제 (브3)-------------#

case = int(input())

for _ in range(case):
    H, W, N = map(int,input().split())
    if N<=W*H:
        print((N%H)*100 + N//H +1)