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
#--실 패--

# ls = [H,M]
#     if M-45<0 and H>0:
#         print(H-1, M+15)

#     elif M-45<0 and H==0:
#         print(H+23,M+15)

#     elif M-45>0 and H==0:
#         print(H,M)

#     elif M-45 > 0:
#         print(H,M)

#     elif M-45 == 0:
#         print(H, M-45)

# if 0 <= H <= 23 and 0 <= M <= 59:



#--실 패--
# if a < 0:
#     print(H-1,a+60)
# elif (a < 0 == H -1) < 0:
#     print(H+)
#     b = H-1
#     if b<0:

# elif a > 0:
#     print(H,a)
# elif H-1<0:
#     print(24-H-1)

# -실패-
# if M-45<0 and H>0:
#     print(H-1, M+15)

# elif M-45<0 and H==0:
#     print(H+23,M+15)

# elif M-45>0 and H==0:
#     print(H,M)

# elif M-45 > 0:
#     print(H,M)

# elif M-45 == 0:
#     print(H, M-45)

# 성공
# if M-45<0 and H>0:
#     H = H-1
#     M = M+15

# elif M-45<0 and H==0:
#     H = H+23
#     M = M+15

# elif M-45>=0:
#     H = H
#     M = M-45

# print(H, M)

#또 다른 정답(선우형)
# h, m = map(int, input().split())
# time = 0
# if h == 0 and m-45 < 0:
#     h = h + 23
#     m = m +15
#     print(h,m)
# else:
#     time = h*60+m
#     alarm = time - 45
#     a_hour = alarm // 60
#     a_minute = alarm % 60
#     print(a_hour, a_minute)


# https://www.acmicpc.net/problem/2558
#------------- A+B - 2 문제 (브5)-------------#
# A = int(input())
# B = int(input())
# print(A+B)
# 0<A, B<10

# https://www.acmicpc.net/problem/5522
#------------- 카드 게임 문제 (브5)-------------#
# a1 = int(input())
# a2 = int(input())
# a3 = int(input())
# a4 = int(input())
# a5 = int(input())
# ls=[a1, a2, a3, a4, a5]
# print(sum(ls))

# https://www.acmicpc.net/problem/5554
#------------- 심부름 가는 길 문제 (브5)-------------#
# S = int(input())
# P = int(input())
# A = int(input())
# H = int(input())
# ls = [S, P, A, H]
# print('%d \n%d' %(sum(ls)/60, sum(ls)%60))

# https://www.acmicpc.net/problem/2739
#------------- 구구단  문제 (브5)-------------#
# N = int(input())

# a = list(range(1, 10))
# for i in a:
#     print('%d * %d = %d' %(N, i, N*i))

# https://www.acmicpc.net/problem/10950
#------------- A+B - 3  문제 (브3)-------------#
# T = int(input())
# for i in range(T):
#     A, B = map(int,input().split())
#     print(A+B)

# https://www.acmicpc.net/problem/15552
#------------- 빠른 A+B  문제 (브2)-------------#
#-정답-#
# import sys
# T = int(sys.stdin.readline())
# for i in range(T):
#     A, B = map(int,sys.stdin.readline().split())
#     print(A+B)
#-실패-
# import sys
# for i in sys.stdin.readline():
#     print(i)

# https://www.acmicpc.net/problem/2741
#------------- N 찍기  문제 (브3)-------------#
# import sys
# N = int(sys.stdin.readline())
# for i in list(range(1,N+1)):
#     print(i)

# 다른예시
# import sys
# N = sys.stdin.readline()
# for i in list(range(1,int(N)+1)):
#     print(i)

# https://www.acmicpc.net/problem/2742
#------------- 기찍 N  문제 (브3)-------------#
# import sys
# N = int(sys.stdin.readline())
# for i in list(range(1,N+1)[::-1]):
#     print(i)
'''
# [::-1] keyworld 차순을 바꿔준다.
# print(list(range(1, 5)[::-1]))  # [4, 3, 2, 1]
'''
# https://www.acmicpc.net/problem/11021
#------------- A+B - 7 문제 (브3)-------------#
# import sys
# T = int(sys.stdin.readline())
# for i in range(T):
#     A,B = map(int,sys.stdin.readline().split())
#     x = i+1
#     a = A+B
#     print(f"Case #{x}: {a}")

# https://www.acmicpc.net/problem/11022
#------------- A+B - 8  문제 (브3)-------------#
# import sys
# T = int(sys.stdin.readline())
# for i in range(T):
#     A,B = map(int,sys.stdin.readline().split())
#     C = A+B
#     x = i+1
#     print(f"Case #{x}: {A} + {B} = {C}")

# https://www.acmicpc.net/problem/2438
#------------- 별 찍기 - 1 문제 (브3)-------------#
# N = int(input())
# for i in range(1, N+1):
#     a = "*"
#     print(a*i)

# a = 'apple'
# b = 'banana'
# c = 2
# print(a*2 )

# https://www.acmicpc.net/problem/2439
#-------------별 찍기 - 2 문제 (브3)-------------#
# N = int(input())
# for i in range(1, N+1):
#     a = "*"*i
#     print(a.rjust(N))

'''
rjust(x) = 오른쪽 '문자열' 정렬, x 에 갯수에 맞춰서 정렬이됨 
ex) 

1.
a = '3'
print(a.rjust(4, '0')) rejust(x, y): x = 칸 갯수 y = 채울곳에 넣을 것 
out :  0003

2.
a = '3'
print(a.rjust(4))
out : '   4' # 4앞에 3칸이 빈칸. 빈칸에 넣어줄 문자를 따로 지정하지 않았기 때문에 빈칸으로 찍힌다.

rjust 뿐 아니라 ljust() = 왼쪽정렬 center() = 가운데등이 있고 zfill() = 기호를 포함해서 인식하며 기호를 맨 앞쪽에 빼주고 빈칸을 0으로 채운다.

zfill ex)
a = '-3'
print(a.zfill(5))
out : -0003  # -까지 포함하여 5칸을 쳐준다.
'''

# https://www.acmicpc.net/problem/10871
#-------------X보다 작은 수 문제 (브3)-------------#

# N, X = map(int,input().split())
# A = map(int,input().split())
# ls = []
# for i in list(A):
#     if i < X:
#         ls.append(i)
# print(*ls)

# *=list 해제
# a = range(1, 5)
# print(list(a))

# https://www.acmicpc.net/problem/10952
#-------------A+B - 5 문제 (브3)-------------#
# import sys
# T = map(int,sys.stdin.readline().split())
# for i in range(1, T+1):
#     A, B = map(int,input().split())
#     C = A+B
#     print(C)
    # while C == 0:
    #     print(C)
    #     break
    # print(A, B)
###정답###
# A, B = map(int,input().split())
# while A+B > 0:
#     print(A+B)
#     A, B = map(int,input().split())
'''while 문에서는 증감식 필수!'''

# https://www.acmicpc.net/problem/10951
#------------A+B - 4 문제 (브3)-------------#
# import sys
# import time
# A, B = map(int,sys.stdin.readline().split())
# time = time.time()
# print(time)
# while time + 5:
    # print(A+B)
    # A, B = map(int,input().split())

# import sys
# A, B = map(int,sys.stdin.readline().split())
# if 0<A<10 and 0<B<10:
#     while True:
#         print(A+B)
#         A, B = map(int,sys.stdin.readline().split())
# exit()

# import sys
# A, B = map(int,sys.stdin.readline().split())
# while 0<A<10 and 0<B<10:
#     print(A+B)
#     A, B = map(int,sys.stdin.readline().split())



# import sys
# A, B = map(int,sys.stdin.readline().split())
# while True:
#     try:
#         0<A<10 and 0<B<10
#         print(A+B)
#         A, B = map(int,sys.stdin.readline().split())
#     except:
#         break

#정답
# import sys
# A, B = map(int,sys.stdin.readline().split())
# while 0<A<10 and 0<B<10:
#     try:
#         print(A+B)
#         A, B = map(int,sys.stdin.readline().split())
#     except:
#         break
'''try except 를 안걸어주면 백준에서 런타임에러가 뜬다. false로 인해 따로 멈추라는 명령어를 바라는듯.'''

# https://www.acmicpc.net/problem/1110
#------------더하기 사이클 문제 (브1)-------------#

# n = int(input())
# c = n
# count=0
# while True:
#     a = c//10
#     b = c%10
#     c = b*10+(a+b)%10
#     count += 1
#     if c == n:
#         break
# print(count)

# 갯수를 뽑기위해 count를 0으로 지정해두고 while문에 +1을 한다
# c = n을 해둔 이유는 while 문 안에 n이 들어가게되면 n에서만 돌기 때문에 첫 수 n에서 c로 변환해주고 그 뒤로 c while을 돌리기 위해 언급.

# https://www.acmicpc.net/problem/10039
#------------평균 점수 문제 (브4)-------------#
# 원섭 = int(input())
# 세희 = int(input())
# 상근 = int(input())
# 숭   = int(input())
# 강수 = int(input())

# student_score = [원섭, 세희, 상근, 숭, 강수]
# study = []
# for i in student_score:
#     # print(average)
#     if i < 40:
#         study.append(40)
#     else:
#         study.append(i)
# average = sum(study)/len(study)

# print(average)

# 정답
# 원섭 = int(input())
# 세희 = int(input())
# 상근 = int(input())
# 숭   = int(input())
# 강수 = int(input())

# student_score = [원섭, 세희, 상근, 숭, 강수]
# study = []
# for i in student_score:
#     # print(average)
#     if i < 40:
#         study.append(40)
#     else:
#         study.append(i)
# average = sum(study)//len(study)
# print(average)

#소수점이 나와서 풀리지 틀렸습니다가 떴다.

# https://www.acmicpc.net/problem/5543
#------------상근날드  문제 (브4)-------------#

# 상덕버거 = int(input())
# 중덕버거 = int(input())
# 하덕버거 = int(input())
# 콜라     = int(input())
# 사이다   = int(input())

# burger = [상덕버거, 중덕버거, 하덕버거]
# beverage = [콜라, 사이다]

# setmenu = min(burger) + min(beverage) - 50
# print(setmenu)

# https://www.acmicpc.net/problem/10817
#------------세 수  문제 (브3)-------------#
# A, B, C = map(int,input().split())
# num = [A, B, C]
# # ans = []
# n_max = max(num)
# n_min = min(num)
# ans = num.remove(n_max)

# print(ans)

# print(ans)
# for i in num:
#     del min(num) and del max
# print()

# a = [1, 2, 3, 4]
# b = []
# c=max(a)
# d=min(a)
# for i in a:
#     if i == max(a) and min(a):
#         a.remove(c)
#         a.remove(d)
# print(*a)
# a.remove()
# print(a)

#정답
# A, B, C = map(int,input().split())
# num = [A, B, C]
# a = max(num)
# b = min(num)
# for i in num:
#     if i == a and b:
#         num.remove(a)
#         num.remove(b)
# print(*num)
# 과연 이게 백준에서 원하는 답이였을까...

# https://www.acmicpc.net/problem/2523 풀다가 말았음
#------------별 찍기 - 13 문제 (브3)-------------#
# n = int(input())
# star = '*'
# a = n
# b = n-1
# print(n*star/n*star)

# while True:
    # print

# n = int(input())
# n = list(range(1, n+1))
# star = '*'
# # print(n)
# for i in n:
#     if i == n:
#         print(i)
# # print(n)

"""n = int(input())

star = []
for i in list(range(1, n+1)):
    a = '*'*i
    star.append(a)
    print(star)"""
    
    # if n == max(range(1, n+1)):
    #     star = '*'*i 

# https://www.acmicpc.net/problem/4299
#------------AFC 윔블던 문제 (브4)-------------#

# 첫째 줄에 두 팀 점수의 합과 차가 빈 칸으로 구분되어 주어진다. 축구 점수는 항상 음이 아닌 정수이고, 합과 차는 1000보다 작거나 같은 음이 아닌 정수이다.
# 첫째 줄에 두 팀의 경기 결과를 출력한다. 득점을 많이 한 쪽을 먼저 출력한다. 만약, 그러한 합과 차를 갖는 경기 결과가 없다면, -1을 출력한다.

# g_dif = int(input())

# g_sum = a+b
# d = (a-b)
# c=5
# d=1
# print(a)
# print(b)
# i = 0
# print(max(g_sum,g_dif), min(g_sum,g_dif))

# a = 0 < g_sum-i < g_sum
# b = g_sum - a
# # i += 1
# # 각팀 a ,b 를 구해주는 프로그램
# g_sum = a+b
# g_dif = a-b>0 #맞다면 프린트 아니면 프린트 -1

# while True:
#     if a + b == g_sum:
#         print(max(a,b))
#         break


# print(a)
# print(b)
# print(g_sum)
# print(g_dif)

# s_max = max(a,b)
# s_min = min(a,b)
# while True:
#     s_max+s_min == g_sum

# s_max = max(a, b)
# s_min = min(a, b)

# 합이 주어질때 a,b를 구해주는 함수
# g_sum = int(input())
# a = 0
# while True:
#     b = g_sum - a
#     (a+b) == g_sum
#     print(a, b)
#     if a > b:
#         break
#     a += 1

# print(a,b)
    # if a > g_sum:
    #     # print(a,b)
    #     break
    # print(a,b)

#헛짓거리

# g_sum = int(input())
# g_dif = int(input())
# a = 0
# while True:
#     b = g_sum - a
#     (a+b) == g_sum
#     (b-a) == g_dif

#     if a > b:
#         break
        
#     else:
#         print(-1)

#     a += 1

# print(a, b)
# 정답 #
# import sys
# g_sum, g_dif = map(int, sys.stdin.readline().split())

# a = g_sum+g_dif
# b = g_sum-g_dif

# team_a = (a)//2  # team a 점수
# team_b = (b)//2  # team b 점수

# mx = max(team_a,team_b)
# mn = min(team_a,team_b)

# if 0<=g_sum<=1000 and 0<=g_dif<=1000:
#     if team_a == team_b:
#         print(mx,mn)

#     elif g_sum < g_dif:
#         print(-1)

#     elif g_dif == 0:
#         print(g_sum, g_dif)

#     elif a%2 and b%2 != 0:
#         print(-1)

#     else:
#         print(mx, mn)


# team_a = (g_sum+g_dif)/2
# team_b = (g_sum-g_dif)/2
# mx = max(team_a,team_b)
# mn = min(team_a,team_b)


# https://www.acmicpc.net/problem/11654
#------------아스키 코드 문제 (브5)-------------#
# n = str(input())
# print(ord(n))

# https://www.acmicpc.net/problem/1076
#------------저항  문제 (브2)-------------#
# a = input().split()
# # b = input().split()
# # c = input().split()
# # a, b, c = input().split()
    

# a = **a
# print(a)
# color = {'black':[0, 1], 
#          'brown':[1, 10],  
#          'red':[2, 100], 
#          'orange':[3, 1000], 
#          'yellow':[4, 10000], 
#          'green':[5, 100000], 
#          'blue':[6, 1000000],
#          'violet':[7, 10000000],
#          'grey':[8, 100000000],
#          'white':[9, 1000000000]}
# # print(color[*a])
# def out_list(a):
    # a = *a
    
# out_list(a)
# printcolor[*a ]
# dict_keys = color.keys()
# dict_values = color.values()
# print(color['+ *a +'])
# print(dict_keys)
# print(dict_values)
# print(a)
# for i in dict_keys:
    # if a == i:
        # print(i)
'''a = list(input().split())
# b = input().split()
# c = input().split()

color = ['black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'grey', 'white']
value = list(range(10))
multi = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

# ju = color,value,multi
# print(ju)
# a = [a]
# b = [b]
# c = [c]

# print(int(str(value[0])+str(multi[0])))
for i in color, value, multi:
    if i == a:
        a = str(value[i])
        print(a)'''
# https://www.acmicpc.net/problem/10797
#------------10부제  문제 (브4)-------------#
# day = int(input())
# a,b,c,d,e = map(int,input().split())

# f = [a,b,c,d,e]
# c = []

# for i in f:
#     if i == day:
#         c.append(i)
# print(c.count(day))



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
# import math
# x, y, cctv = map(int,input().split())

# a = math.ceil(x/cctv)
# b = math.ceil(y/cctv)

# print(a*b)

# https://www.acmicpc.net/problem/16199
#-----------나이 계산하기 문제 (브4)-------------#
birth = list(map(int,input().split()))
standar = list(map(int,input().split()))

year = standar[0] - birth[0]
month = standar[1] - birth[1]
date = standar[2] - birth[2]

# if birth[0] == standar[0]:
#     print(0)
if year > 0 and month >= 0 and date >= 0:
    print(year)
else:
    print(0)

count_age = year + 1
year_age = year

print(count_age)
print(year_age)



# https://www.acmicpc.net/problem/19532
#----------수학은 비대면강의입니다  문제 (브3)-------------#



# 정답

# a,b,c,d,e,f = map(int,input().split())

# x = (c*e-b*f)/(a*e-b*d)
# y = (a*f-c*d)/(a*e-b*d)
# print(int(x),int(y))



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

JOI군은 내일부터 새로운 소셜 게임을 시작하기로 결정했다.

이 소셜 게임에서는 하루에 대하여. 1회까지 로그인할 수 있으며, 로그인할 때마다 우리는 A장의 동전을 얻을 수 있다.

또, 월요일부터 일요일까지 7일 연속 로그인을 하면 그때마다 추가적으로 비 개의 동전을 얻을 수 있다.

코인을 이것 말고는 다른 코인을 받을 수 없다.

내일은 월요일이다. JOI가 적어도 C장의 동전을 얻기 위해 로그인해야 하는 최소 횟수를 구하라.