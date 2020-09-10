# data.lower() 가 어떻게 표시되나 확인 해보기 위한 코드

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

data = "In the town Athy one Jeremy Lanigan \n Battered away"
corpus = data.lower().split("\n")

print(corpus)
# ['in the town athy one jeremy lanigan ', ' battered away'] 소문자로 구분되서 나온다. 

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1