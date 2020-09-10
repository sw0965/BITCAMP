# https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home
# is_sarcastic: 1 if the record is sarcastic other 0

import json
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 100
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
TRAINING_SIZE = 20000

# unzip file
# local_zip = 'SELF_TENSOR/NLP/news-headlines-dataset/30764_533474_bundle_archive.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('SELF_TENSOR/NLP/news-headlines-dataset/data')
# zip_ref.close()

with open ("SELF_TENSOR/NLP/news-headlines-dataset/data/Sarcasm_Headlines_Dataset_v2.json", 'r') as f:
    datastore = json.load(f)

print("hello")