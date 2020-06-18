import os, glob, numpy as np
from PIL import Image


x_predict = []

imge_dir = 'D:/과일/train/test'
file = glob.glob(imge_dir + '/*.jpg')
file.sort()
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