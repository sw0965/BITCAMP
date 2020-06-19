from PIL import Image
import pandas as pd
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
from sklearn.preprocessing import StandardScaler


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
    # apple  파일 길이 :  40
    # banana  파일 길이 :  40
    # grape  파일 길이 :  40
    # pineapple  파일 길이 :  40

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.array(img)
        # print(data.shape)
        # print(data)
        # print(type(data)) #numpy 타입
        x.append(data)
        y.append(label)
        # print('x : ',type(x)) #list 타입으로 변환되었다.
        # print('y : ',type(y)) #list 타입으로 변환되었다.


x = np.array(x)  # numpy 형식으로 변환
y = np.array(y)  # numpy 형식으로 변환
# print(x.shape)
# print(y.shape)
#enumerate = 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 수 있습니다. 이때 사용합니다. 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환합니다.




# x_predict 만들기
x_predict = []

imge_dir = 'D:/과일/train/test'
file = glob.glob(imge_dir + '/*.jpg')
# print(file)
for i in file:
    img = Image.open(i)
    img = img.resize((100, 100))
    # print(img)
    data = np.array(img)
    # print(data)
    x_predict.append(data)
    # print('x_type : ', type(x_predict))  #list 형태

x_predict = np.array(x_predict)
# print('type : ', type(x_predict))     # numpy 형태




# train test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
# print(x_train.shape) #(128, 100, 100, 3)
# print(x_test.shape)  #(32, 100, 100, 3)
# print(y_train.shape) #(128 , 4)
# print(y_test.shape)  #(32 , 4)

# 스케일러 위해 리쉐이프
x_train = x_train.reshape(128, 30000)
x_test = x_test.reshape(32, 30000)
x_predict = x_predict.reshape(20, 30000)

x_train = x_train/255
x_test = x_test/255
x_predict = x_predict/255

# 스케일러
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_predict = scaler.transform(x_predict)


# 모델링을 위해 다시 리쉐이프
x_train = x_train.reshape(128, 100, 100, 3)
x_test = x_test.reshape(32, 100, 100, 3)
x_predict = x_predict.reshape(20, 100, 100, 3)


np.save('./mini_project/npy_data/x_predict_test1.npy', x_predict)
np.save('./mini_project/npy_data/x_train_test1.npy', x_train)
np.save('./mini_project/npy_data/x_test_test1.npy', x_test)
np.save('./mini_project/npy_data/y_train_test1.npy', y_train)
np.save('./mini_project/npy_data/y_test_test1.npy', y_test)


