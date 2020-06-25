# https://www.acmicpc.net/problem/1001
#------------- A-B 문제 (브5)-------------#
# A, B = map(int,input().split())
# print(A-B)


# https://www.acmicpc.net/problem/10171
#------------- 고양이 문제 (브5)-------------#
# print("\    /\\")
# print(" )  ( ')")
# print("(  /  )")
# print(" \(__)|")

# https://www.acmicpc.net/problem/10998
#------------- A*B 문제 (브5)-------------#
# A, B = map(int,input().split())
# print(A*B)

# https://www.acmicpc.net/problem/10869
#------------- 사칙연산 문제 (브5)-------------#
# A, B = map(int, input().split())
# print(A+B)
# print(A-B)
# print(A*B)
# print('%d'%(A/B))
# print(A%B)

# https://www.acmicpc.net/problem/2588
#------------- 곱셈 문제 (브4)-------------#   
# A = int(input())
# B = input()                  # B를 스트링으로 납둔 상태에서 연산때 int로 변환
# print(A*int(B[2]))
# print(A*int(B[1]))
# print(A*int(B[0]))
# print(A*int(B))
'''
# A = '5'
# b = 2
# print(int(A)*b)
# print(int(A[0]))
'''
# https://www.acmicpc.net/problem/1330
#------------- 두수 비교하기 문제 (브4)-------------#
# A, B = map(int,input().split())
# if A > B:
#     print('>')
# elif A < B:
#     print('<')
# elif A == B:
#     print('==')

# https://www.acmicpc.net/problem/9498
#------------- 시험성적 문제 (브4)-------------#
# result = int(input())
# if 90 <= result <= 100:
#     print('A')
# elif 80 <= result <= 89:
#     print('B')
# elif 70<=result<=79:
#     print('C')
# elif 60<=result<=69:
#     print('D')
# else:
#     print('F')

# https://www.acmicpc.net/problem/2753
#------------- 윤년 문제 (브4)-------------#
# year = int(input())
# # year = 1<= year <= 4000
# if year == year%4 or year%100:
#     print(1)
# elif year == year%400:
#     print(1)
# else:
#     print(0)

# year = int(input())
# # year = 1<= year <= 4000
# if 0 == year%4 or year%100:
#     print(1)
# elif 0 == year%400:
#     print(1)
# else:
#     print(0)
################정답################################
# year = int(input())
# if year%4 == 0 and not(year%100 == 0) or year%400==0:
#     print(1)
# else:
#     print(0)
# year = 1<= year <= 4000  # 범위 넣어주니까 틀림 ㅡㅡ
# not을 쓸때는 ()가 필요하다.


# https://www.acmicpc.net/problem/14681
#------------- 사분면 고르기 문제 (브4)-------------#
# x = int(input())
# y = int(input())
# if x > 0 and y > 0:
#     print(1)
# elif x < 0 and y > 0:
#     print(2)
# elif x < 0 and y < 0:
#     print(3)
# elif x > 0 and y < 0:
#     print(4)

# https://www.acmicpc.net/problem/2884
#------------- 알람 시계 문제 (브3)-------------#
# import datetime
# H, M = map(int,input().split())
# if 0 <= H <= 23 and 0 <= M <= 59:
#     if :


# https://www.acmicpc.net/problem/2558
#------------- A+B - 2 문제 (브5)-------------#
# A = int(input())
# B = int(input())
# print(A+B)
# 0<A, B<10

# MIN, MAX = map(int,input().split())
# if 1 <= MIN <= 1000000000000 and MIN <= MAX <= MIN+1000000:
