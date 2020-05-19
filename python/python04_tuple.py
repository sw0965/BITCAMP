#2. 튜플
#리스트와 거의 같으나, 삭제, 수정이 안된다.
a = (1,2,3)
b = 1, 2, 3
print(type(a))
print(type(b))

#a.remove(2)
#print(a)

print(a + b)    #list 처럼 붙어서 나온다. 안에 내용은 수정이 안되므로.
print(a * 3)

# 튜플은 잘 안쓰고 list를 가장 많이 사용.

#print(a - 3)   #수정이 안되므로 불가능.