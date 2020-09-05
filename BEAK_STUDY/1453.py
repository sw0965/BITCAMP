# https://www.acmicpc.net/problem/1453 피시방 알바
#-----------  피시방 알바 문제 (브1)-------------#

# customer = int(input())

# a = range(customer)
# print(a)
# for _ in range(customer):
#     seat = 
# seats = map(int,input().split())

# pc = []
# for i in seats:
#     if i in seats:
#         i == i
#         print(i)
        # pc.append(i)
        # print(i)
    # pc.append(i)
    # print(pc)
"""cnt = []
count = {}
for i in seats:
    pc.append(i)

for i in pc:
    try: count[i] += 1
    except: count[i] = 1
a = set(count.values())

for i in a:
    if i >= 2:
        cnt.append(i)
        print(cnt)
알고리즘 복잡하게 가다가 수학적으로 생각하기로 함."""
# print(sum(a))
# cnt.append(a)
# print(cnt)

    # set
    # if dict.values() <= 2:
        # print(count.values())

# print(count.keys())
# print(count.values())

"""----------정------답---------"""
customer = int(input()) # 손님 수

seats = map(int,input().split()) # 원하는 좌석

pc = [] # 원하는 pc 
cnt = {} # 중복된 값들
for i in seats:
    '''손님들이 원하는 좌석들을 pc라는 list에 넣기'''
    pc.append(i)

for i in pc:
    '''cnt라는 딕셔너리를 만들어서 key값을 뽑기위한 작업
    만약 5명 손님이 1 1 2 2 3 이라는 pc 를 원할때
    1:2 2:2 3:1 이렇게 key값이 3개 나오기 때문에
    전체 손님수에서 key값을 빼준다.'''
    try: cnt[i] += 1
    except: cnt[i] = 1
# print(set(cnt.keys))
a = cnt.keys()
print(customer - len(a)) # 전체 손님수에서 key값을 뺀다.
# print(a)