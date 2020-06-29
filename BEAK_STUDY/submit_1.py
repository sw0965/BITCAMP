# https://www.acmicpc.net/problem/1330
# ------------- 두수 비교하기 문제 (브4)-------------#
A, B = map(int,input().split())
if A > B:
    print('>')
elif A < B:
    print('<')
elif A == B:
    print('==')