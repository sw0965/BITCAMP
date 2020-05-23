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
IndentationError: expected an indented block
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
from re import * # 이런! re에도 match라는 함수가 존재한다. 
print(match)     #"<function match at 0x10281e6a8>"