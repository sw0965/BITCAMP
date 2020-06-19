# 자료형
#1. 리스트

a = [1,2,3,4,5]
b = [1,2,3,'a','b']
print(b)
#numpy는 딱 한가지 자료형만 사용할 수 있다.

print(a[0] + a[3])
#print(b[0] + b[3])
print(type(a))
print(a[-2])
print(a[1:3])

a = [1,2,3, ['a','b','c']]
print(a[1])
print(a[-1])
print(a[-1][1]) # 결과값 = b

#1-2. 리스트 슬라이싱

a = [1,2,3,4,5]
print(a[:2])

#1-2. 리스트 더하기

a = [1,2,3]
b = [4,5,6]
print(a + b)

c = [7,8,9,10]
print(a + c)

print(a * 3)

#print(a[2] + 'hi')
print(str(a[2]) + 'hi')

f = '5'
#print(a[2] + f)
print(a[2] + int(f))

#리스트 관련 함수

a = [1,2,3]
a.append(4)        #append 덧붙이다. (완전중요)
print(a)
print(a)           # a = a.append(5) 오류 : # a = 쓰면 안된다.



a = [1,3,4,2]
a.sort()           #sort 정렬
print(a)


a.reverse()        #reverse 역순,거꾸로
print(a)


print(a.index(3))  # == a[3]
print(a.index(1))  # == a[1]


a.insert(0, 7)     #[7, 4, 3, 2, 1] 삽입하는거지 바꾸는게 아님
print(a)
a.insert(3, 3)
print(a)


a.remove(7)        # [4, 3, 2, 1]   안에 있는 값을 지움.
print(a)
a.remove(3)        # 3이 두개있을때 remove 3을 쓰면 먼저 걸리는 수가 지워짐.
print(a)