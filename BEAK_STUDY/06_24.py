'''
about input- 데이터를 저장하고 가공하기 위해서는 데이터에 대한 입력이 필요하다. input은 사용자에게 데이터를 입력 받기 위한 함수입니다. 
             Input은 입력 받은 값을 문자열로 처리하기 때문에 숫자를 입력받을 경우 숫자형 Type으로 변환해주어야 합니다.

#------------- 1번 문제 -------------#

A, B = input().split()  # input에 a ,b 를 저장해주고 +을 해줄때 두 수를 나눠 써주어야(띄어쓰기) 하기 때문에 split 사용
print(int(A)+int(B))    # A와 B는 문자이기 때문에 int로 숫자화

#------------- 2번 문제 -------------#

n = int(input())              # n = 숫자형을 저장해주고
def loop(N):
    a = 0                     # a 는 0이다    
    for n in range(1, N+1):   # range 1부터 N+1(제로 베이스라 숫자 5를 원하면 6을 입력해야되기 때문에 +1)의 순차적으로 n이 들어갈때
        a += n                # a+n=a 
    return a                  # a 를 돌려준다.
print(loop(n))

#------------- 3번 문제 -------------#

n, m = input().split() # n과 m을 저장 해주고 split으로 나눠 쓸 수 있게 해준다.
print(int(n)//int(m))  # %d를 썻는데 런타임 오류 파이썬은 기본이 float 형태라 int로 변환시켜줘야함 그냥 /나누기 했을경운 float으로 뜨기 때문에 //로 사용
print(int(n)%int(m))   # % = 나머지 값을 구할때 사용한다 int형 일때 10/3 = 3 , 나머지 1 

# 런타임 에러 #

n, m = input().split()
print('%d\n%d' %(int(n)/int(m),int(n)%int(m)))

print(int(n)/int(m))
print(int(n)%int(m))

'''

# a , b = int(input("두 숫자를 입력해 주세요: ").split())  #  에러 소스
# print('두 숫자의 합은 :', int(a+b), '입니다.')
# print('두 숫자의 나눗셈은 :', int(a/b), '입니다.')


# a , b = map(int,input("두 숫자를 입력해 주세요: ").split())  # map은  요소를 지정된 함수로 처리해주는 함수입니다
# print('두 숫자의 합은 :', int(a+b), '입니다.')
# print('두 숫자의 나눗셈은 :', int(a/b), '입니다')

# 정수로 변환 시켜주기 #

a = [1.1, 2.2, 3.3, 4.4, 5.5]

for i in range(len(a)):
    a[i] = int(a[i])
print(a)

# map 사용해서 한줄로 편하게
print(list(map(int, a)))

