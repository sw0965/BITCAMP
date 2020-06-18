from PIL import Image
import pandas as pd
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir


imge_dir = 'D:/과일/train'
categories = ['apple', 'banana', 'grape', 'pineapple']
nb_classes = len(categories)

image_w = 100
image_h = 100
pixels = image_h * image_w * 3

### 이미지 파일 Data화
x = []
y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1    
    image_dir = imge_dir + '/' + cat
    files = glob.glob(image_dir + "/*.jpg")
    print(cat, " 파일 길이 : ", len(files))

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.array(img)
        x.append(data)
        y.append(label)

x = np.array(x)  # numpy 형식으로 변환
y = np.array(y)  # numpy 형식으로 변환
print(x.shape)
print(y.shape)

#enumerate = 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 수 있습니다. 이때 사용합니다. 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.




# x_predict 만들기
x_predict = []

imge_dir = 'D:/과일/train/test'
file = glob.glob(imge_dir + '/*.jpg')
print(file)
for i in file:
    img = Image.open(i)
    img = img.resize((100, 100))
    print(img)
    data = np.array(img)
    print(type(data))
    x_predict.append(data)
    print('x_type : ', type(x_predict))  

x_predict = np.array(x_predict)
print('type : ', type(x_predict))     

np.save('./data/mini_project/x_data.npy', x)
np.save('./data/mini_project/y_data.npy', y)
np.save('./data/mini_project/x_predict.npy', x_predict)

