'''
# for 문 continue ( break와 달리 특정 조건일때 루프를 건너 뜀. )
A = [1, 2, 3]
for n in A:
    if n == 2:
        continue
    print(n)

B = [1, 2, 3, 4, 5, 6]
for n in B:
    if n % 2 == 0:
        continue
    print(n)

# For 문에서 index 표시
list = ['a', 'b']
for index, value in enumerate(list):
    print(index,value)

list = ['tiger', 'dog', 'elephant']
for index, value in enumerate(list):
    print(index,value)
'''
# 연습문제 

items = {"지우개":[100, 2], "펜":[200,3], "노트":[400,5]}
total_price = 0

for item in items:
    print(item + '은(는) 한 개에' + str(items[item][0]) + '원이며, ' + str(items[item][1]) + '개 구입합니다.')
    total_price = items[item][0]*items[item][1]
    print('지불해야 할 금액은 ' + str(total_price) + '원입니다.')

    money = 1500
    if total_price < money:
        print('거스름 돈은 ' + str(money - total_price) +'입니다')
    elif money == total_price:
        print('거스름돈은 없습니다.')
    else:
        print('돈이 부족합니다.')
    print('')


city = "Seoul"
big = city.upper()
# print(city.upper()) #프린트 내에서만 바뀜 
print(big)

# def cube_cal(n):
#     print(n ** 3)
# cube_cal(4)

# def introduce(name, age):
#     print(name+"입니다. "+age+"살입니다.")
# introduce('홍길동','18')


# def introduce(first = '김', second = '길동'):
    # print('성은' +first +'이고, 이름은' +second +'입니다.')
# introduce('홍','방구')

class Myproduct:
    def __init__(self, name, price, stock): #여기 갯수랑
        self.name = name
        self.price = price
        self.stock = stock
        self.sale = 0
product1 = Myproduct("cake", 500, 20)   # 여기 갯수랑 맞춰줘야됌
print(product1.stock) 

class Mypro:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
# 구매 매서드
def buy_up(self, n):
    self.stock += n 
# 판매 매서드
def sell(self,n):
    self.stock -= n
    self.sales += n*self.price
# 개요 매서드
def summary(self):
    message = "called summary().\n name:" + self.name + \
    "\n pirce: " + str(self.price) + \
    "\n stock: " + str(self.stock) + \
    "\n sales: " + str(self.sales) 
    print(message) 

import time
# start = time.time() # 시간 체크하는 기능.

import numpy as np 
# arr = np.arange(10).reshape(2,5)
# print(arr.T)
# print(np.transpose(arr))

# arr = np.array([15, 30, 5])   # sort 하면 순서대로 5 15 30 이 되고 그걸 인덱스로 변형 해주면 5 = 2, 15 = 0, 30 = 1이 되므로 201이 됌
# print(np.argsort(arr))

# arr = np.array([[8, 4, 2], [3, 5, 1]])
# print(np.argsort(arr))

# arr = np.sort(arr)
# print(arr)

arr = np.array([[8, 4, 2], [3, 5, 1]])
arr.sort(0)  # 0 은 열 정렬, 1은 행 정렬
print(arr)
print("\n")

#pandas 의 Series
import pandas as pd 
# index = ["apple", "orange", "banana", "strawberry"]
# data = [10, 5, 7, 4]
# series = pd.Series(data, index=index)
# print(series)
# item = series[0:3]
# print(item)
# print (series[0:3])

f = {"banana":3, "orange":4, "grape":1, "peach":5}
s = pd.Series(f)
print(s[0:2])
# series.append 와 series.drop을 통해 추가 삭제 가능

np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
# DataFrame 을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11),10)

df.index = range(1, 11)
print(df)

df = df.loc[[2,3,4,5],["banana", "kiwifruit"]]
print(df)

data = {"fruit": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

print(df)
print()
print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])

index = ["growth", "mission", "ishikawa", "pro"]
data = [50, 7, 26, 1]

series = pd.Series(data, index=index)
print(series)
aidemy = series.sort_index()
print(aidemy)
print()
aidemy1 = pd.Series(['30'], ['tutor'])
aidemy2 = series.append(aidemy1)
print(aidemy2)

df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1,11),10)
df.index = range(1, 11)
print(df)
aidemy3 = df.loc[[2,3,4,5],['ishikawa']]
print()
print(aidemy3)

# 데이터 프레임 연결
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1,101), len(index))
    df.index = index
    return df

print(np.arange(1,11,2))