# 코딩을 잘하는 조건 조건문과 반복문.
# for문은 반복문

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

# 파이썬에서 from for 이런거에서 엔터쳤을때 앞에 띄어쓰기가 생기는이유는 그곳에 포함되 있다는 뜻

for i in a.keys():
    print(i)

a = [1,2,3,4,5,6,7,8,10]
for i in a:
    i = i*i
    print(i)
    # print('hi')

for i in a:
    print(i)

## while문
'''
while 조건문 :   # 참(true)일 동안 계속 돈다.
    수행할 문장
'''

### if문

if 1 : 
    print('true')
else : 
    print('false')

if 3 : 
    print('true')
else : 
    print('false')

if 0 : 
    print('true')
else : 
    print('false')

if -1 : 
    print('true')
else : 
    print('false')

'''
비교연산자
<, >, ==, !=, >=, <=
'''

a = 1
if a == 1:
    print('출력잘되')

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')    

### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우먹자')
else:
    print('라면먹자')    

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")

############################
# break, continue
print('==================break================')
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")

print('==================continue================')
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:                             #break든 continue든 if 이하에 데이터값에 따라 다시 for 문으로 돌아감
    if i < 30:
        continue
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")


jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:                           
    if i < 60:
        continue
    if i >= 60:
        print("경] 합격 [축")
        number = number + 1

print("합격인원 :", number, "명")