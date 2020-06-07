# 2.4 들여쓰기

# '#' 기호는 주석을 의미한다. 
# 파이썬에서 주석은 실행되지 않지만, 코드를 이해하는데 도움이 된다. 
for i in [1,2,3,4,5]:
    print(i)              # 'for i' 단락의 첫 번째 줄
    for j in [1,2,3,4,5]:
        print(j)          # 'for j' 단락의 첫 번째 줄
        print(i+j)        # 'for j' 단락의 마지막 줄
        print(i)             # 'for i' 단락의 마지막 줄
print("dome looping")

'''공백문자는 소괄호() 와 대괄호[] 안에 서는 무시되기 때문에 다음과 같은 긴 계산을 하거나'''
long_winded_computation = (1 +2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)
''' 코드의 가독성을 높이는데 유용하게 쓸 수 있다.'''
list_of_lists = [[1,2,3], [4,5,6], [7,8,9]]
easier_to_read_list_of_lists = [[1,2,3],
                                [4,5,6],
                                [7,8,9]]

'''역슬래시를 사용하면 코드가 다음 줄로 이어지는 것을 명시할 수 있다.'''
two_plus_three = 2 + \
                 3
'''들여쓰기를 사용함으로써 생기는 한 가지 문제는 코드를 복사해서 파이썬 셸에
붙여넣을 때 어려움을 겪을 수 있다는 것이다. 예를 들어 다음과 같은 코드를 파이썬 셸에 붙여넣기 하면'''
for i in [1, 2, 3, 4, 5]:

    # 빈 줄이 있다는 것을 확인하자.
    print(i)

''' 인터프리터가 빈 줄을 보고 for 문이 끝난 것으로 판단해서 다음과 같은 에러가 출력될 것이다.'''
# IndentationError: expected an indented block
''' 한편 IPython 에는 %paste라는 특별한 명령어가 있어서 공백 문자뿐만 아니라
클립보드에 있는 무엇이든 제대로 붙여넣을 수 있다. 이것 하나만으로도 IPython을 쓸 이유는 충분하다.'''

# 모듈
'''모듈을 사용하기 위해선 import를 사용해야된다.'''
import re
my_regex = re.compile("[0-9]+", re.I)
'''여기서 불러온 re는 정규표현식을 다룰 때 필요한 다양한 함수와 상수를 포함. 
그 함수와 상수를 사용하기 위해서는 re 다음에 마침표를 붙인 후 함수나 상수의 이름을 이어서 쓰면 된다.'''
import re as regex 
my_regex = regex.compile("[0-9]+", regex.I)

'''모듈의 이름이 복잡하거나 이름을 반복적으로 타이핑할 경우에도 별칭을 사용할 수 있다. 
예를 들어 matplotlib 라는 라이브러리로 데이터를 시각화할 때는 다음과 같은 별칭을 관습적으로 사용한다.'''
import matplotlib.pyplot as plt

plt.plot(...)

'''모듈 하나에서 몇몇 특정 기능만 쓸 수 있다'''
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

'''가장 좋지 않은 습관 중 하나는 모듈의 기능을 통째로 불러와서 기존의 변수들을 덮어 쓰는 것이다.'''
match = 10
# from re import * # 이런! re에도 match라는 함수가 존재한다. 
# print(match)     #"<function match at 0x10281e6a8>"

# 2.6 함수

'''파이썬에선 def를 이용해 함수를 정의'''
def double(x):
    return x *2

'''파이썬 함수들은 변수로 할당되거나 함수의 인자로 전달할 수 있다는 점에서 일급 함수의 특성을 가진다'''
def apply_to_one(f):
    '''인자가 1인 함수 f를 호출'''
    return f(1)

my_double = double             # 방금 정의한 함수를 나타낸다.
x = apply_to_one(my_double)    # 2

'''짧은 익명의 람다 함수도 간편하게 만들 수 있다.'''

y = apply_to_one(lambda x: x + 4)  #5

'''대부분의 사람들은 그냥 def를 사용하라고 얘기하겠지만, 변수에 람다 함수를 할당할 수도 있다.'''

another_double = lambda x: 2 * x   # 이 방법은 최대한 피하도록 하자.

# def another_double(x):
    # '''대신 이렇게 작성하자.'''
    # return 2 * x

'''함수의 인자에는 기본값을 할당할 수 있는데, 기본값 외의 값을 전달하고 싶을때는 값을 직접 명시해 주면 된다.'''

def my_print(message = "my default message"):
    print(message)

my_print("hello") # 'hello'를 출력
my_print()        # 'my default message'를 출력

'''가끔 인자의 이름을 명시해 주면 편리하다.'''

def full_name(first = "What's-his-name", last = "Something"):
    return first + "" + last

full_name("Joel", "Grus")   # 'Joel Grus'를 출력
full_name("Joel")           # 'Joel Something'을 출력
full_name(last="Grus")      #'What's-his-name Grus'를 출력

# 2.7 문자열 

'''문자열은 작은 따옴표 또는 큰 따옴표로 묶어 나타낸다. (다만, 앞 뒤로 동일한 기호를 사용)'''
single_quoted_string = 'data science'
double_quoted_string = "data science"

'''파이썬은 몇몇 특수 문자를 인코딩때 역슬래시 한다.'''
tab_string = "\t"   # 탭을 의미하는 문자열
len(tab_string)     # 1

'''만약 역슬래시를 역슬래시로 보이는 문자로 사용하고 싶다면
(특히 윈도우 디렉터리 이름이나 정규표현식에서 사용하고 싶을 때) 문자열 앞에
r을 붙여 rawstring(가공되지 않은 문자열) 이라고 명시하면 된다.'''
not_tab_string = r"\t"   # 문자 '\'와 't'를 나타내는 문자열
len(not_tab_string)      # 2 

'''세 개의 따옴표를 사용하면 하나의 문자열을 여러 줄로 나눠서 나타낼 수 있다.'''
multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""

'''파이썬 3.6부터 문자열 안에 값을 손쉽게 추가할 수 있는 f-string 기능이 추가되었다. '''
first_name = "Jeol"
last_name = "Grus"

'''full_name 변수 만들기'''
full_name1 =first_name + "" + last_name   # 문자열 합치기
full_name2 = "{0} {1}".format(first_name, last_name) # .format을 통한 문자열 합치기
'''f-string 사용해서 합치기'''
full_name3 = f"{first_name} {last_name}"

# 2.8 예외 처리

'''예외가 발생했음을 알려준다.'''
try:
    print(0 / 0)
except ZeroDivisionError:
    print("cannot divide by zero")

# 2.9 리스트        
''' 파이썬 기본적인 데이터 구조는 리스트이다. 리스트는 순서가 있는 자료의 집합이다.
(다른 언어에선 array(배열) 이라고 하는 것과 유사하지만, 리스트의 기능이 조금 더 풍부하다.)'''
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]

list_length = len(integer_list) # 결과는 3
list_sum =    sum(integer_list) # 결과는 6

'''대괄호를 사용해 리스트의 n번째 값을 불러오거나 설정할 수 있다.'''
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
zero = x[0]  # 결과는 0, 리스트의 순서는 0부터 시작한다.
one = x[1]
nine = x[-1]
eight = x[-2]
x[0] = -1  # x는 이제 [-1,1,2,3, ..., 9]

'''슬라이싱'''
first_three = x[:3]   #-1, 1, 2
three_to_end = x[3:]  # 3, 4, ..., 9
one_to_four = x[1:5]  # 1, 2, 3, 4 
last_three = x[-3:]   # 7, 8, 9
without_first_and_last = x[1:-1] # 1, 2, ..., 8
copy_of_x = x[:]      # -1, 1, 2, ..., 9

'''간격 설정'''
every_third = x[::3] # -1, 3, 6, 9
five_to_three = x[5:2:-1] #5, 4, 3

'''항목 존재 여부 확인'''
1 in [1,2,3]  #True
0 in [1,2,3]  #False

'''리스트 추가'''
x = [1, 2, 3]
x.extend([4, 5, 6])   # x 는 이제 [1, 2, 3, 4, 5, 6]

x = [1,2,3]
y = x = [4,5,6]
# 25p
