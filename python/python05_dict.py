#3. 딕셔너리 # 중복 x
# {키 : 벨류}
# {key : value}

a = {1: 'hi', 2: 'hello'}
print(a)
print(a[1])

b = {'hi' : 1, 'hello': 2}
print(b['hello'])

# 딕셔너리 요소 삭제
del a[1]                          # dictionary 같은 경우에 []에 키 값이 들어간다.
print(a)
del a[2]
print(a)

a = {1:'a', 2:'b', 1:'b', 1:'c'}  # 키 중복일땐 마지막 덮어쓴것만 나온다.
print(a)

b = {1:'a', 2:'b', 3:'a'}         # value가 중복일땐 다 출력된다. (시험에서 1번 100점, 2번 100점, 3번 100점 같은 것.)
print(b)

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())         # dict_keys(['name', 'phone', 'birth'])
print(a.values())       # dict_values(['name', 'phone', 'birth'])
print(type(a))          # class 'dict'
print(a.get('name'))    # yun
print(a['name'])        # yun
print(a.get('phone'))   # 010
print(a['phone'])       # 010
