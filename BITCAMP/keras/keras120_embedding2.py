from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요", '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', 
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print("token.word_index: \n", token.word_index)
# token.word_index:
#  {'너무': 1, '참': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에
# 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}
# 중복 제거된 인덱싱. 


# '참'이라는 단어를 3번 주면?
# 많이 사용하는 단어가 인덱싱 우선순위가 됨
# token.word_index:
#  {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에
# 요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}

x = token.texts_to_sequences(docs)
print("token.texts_to_sequences: \n", x)
# 문자를 수치화
# token.texts_to_sequences:
#  [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], 
# [2, 22], [1, 23]]

# 문제점? shape가 동일 하지 않는 점
# shape를 맞춰줘야함. 하나하나를 reshape 할 수 없음
# pad_sequences를 써준다면! padding을 쓰면 빈자리에 0을 넣어서 진행
# 제일 큰 shape의 숫자를 맞춰서 나머지는 0으로 채우면 동일한 shape로 됨
# LSTM의 경우 : 의미 있는 인덱싱이 뒤로 가는 것이 좋을 수 있음


from keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre')
print("pad_sequences_pre: \n", pad_x)
# pad_sequences_pre:
#  [[ 0  0  2  3]
#  [ 0  0  1  4]
#  [ 1  5  6  7]
#  [ 0  8  9 10]
#  [11 12 13 14]
#  [ 0  0  0 15]
#  [ 0  0  0 16]
#  [ 0  0 17 18]
#  [ 0  0 19 20]
#  [ 0  0  0 21]
#  [ 0  0  2 22]
#  [ 0  0  1 23]]
# padding='pre' 앞에서부터 0을 채워준다.

pad_x = pad_sequences(x, padding='post')
print("pad_sequences_post: \n", pad_x)
# pad_sequences_post:
#  [[ 2  3  0  0]
#  [ 1  4  0  0]
#  [ 1  5  6  7]
#  [ 8  9 10  0]
#  [11 12 13 14]
#  [15  0  0  0]
#  [16  0  0  0]
#  [17 18  0  0]
#  [19 20  0  0]
#  [21  0  0  0]
#  [ 2 22  0  0]
#  [ 1 23  0  0]]
# padding='post' 뒤에서부터 0을 채워준다.

pad_x = pad_sequences(x, value=1.0)
print("pad_sequences_value: \n", pad_x)
# pad_sequences_value:
#  [[ 1  1  2  3]
#  [ 1  1  1  4]
#  [ 1  5  6  7]
#  [ 1  8  9 10]
#  [11 12 13 14]
#  [ 1  1  1 15]
#  [ 1  1  1 16]
#  [ 1  1 17 18]
#  [ 1  1 19 20]
#  [ 1  1  1 21]
#  [ 1  1  2 22]
#  [ 1  1  1 23]]
# value=1.0는 0이 아닌 value 값으로 채워짐


''' 명석이 소스
from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밋어요", "최고에요", "참 잘 만든 영화에요",
        '추천하고 싶은 영화입니다', '한번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밋네요']

# 긍정 1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])


# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index) # 중복 된 단어들은 앞 쪽으로 몰림 그리고 한번만 등장(인덱스 번호니까) => 많이 등장하는 놈이 맨 앞으로

x = token.texts_to_sequences(docs)
# print(x)

from keras.preprocessing.sequence import pad_sequences
# 패드 시퀀스 
'''
'''
(2,) [3,7]
(1,) [2]
(3,) [4,5,11]
(5,) [5,4,3,2,6]
'''
'''

pad_x_pre = pad_sequences(x, padding = 'pre')
print(pad_x_pre)

pad_x = pad_sequences(x, padding = 'post',value = 1.0)
print(pad_x)
'''