# https://www.acmicpc.net/problem/2753
#------------- 윤년 문제 (브4)-------------#

year = int(input())
if year%4 == 0 and not(year%100 == 0) or year%400==0:
    print(1)
else:
    print(0)
# year = 1<= year <= 4000  # 범위 넣어주니까 틀림 ㅡㅡ
# not을 쓸때는 ()가 필요하다.