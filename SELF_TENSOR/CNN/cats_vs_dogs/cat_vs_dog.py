import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd 

# 현재 작업경로 확인 = os.getcwd()
# 작업경로 안에 들어있는 파일 리스트 확인하기 = os.listdir(path)

# Unzip Data.
# local_zip = 'SELF_TENSOR/cats_vs_dogs_data/dogs-vs-cats.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('SELF_TENSOR/cats_vs_dogs_data')
# zip_ref.close()

# local_zip = 'SELF_TENSOR/cats_vs_dogs_data/test1.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('SELF_TENSOR/cats_vs_dogs_data')
# zip_ref.close()

# local_zip = 'SELF_TENSOR/cats_vs_dogs_data/train.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('SELF_TENSOR/cats_vs_dogs_data')
# zip_ref.close()

print(len(os.listdir('SELF_TENSOR/cats_vs_dogs_data/train/')))  # 25000
print(len(os.listdir('SELF_TENSOR/cats_vs_dogs_data/test1/')))  # 12500

