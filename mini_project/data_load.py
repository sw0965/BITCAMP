from PIL import Image
import pandas as pd
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )

a = [4, 5, 6, 7]
for i in a:
    print(i)

for i in range(4, 8):
    print(i)

fa = ['yang', 'snagwoo', 'jongsu', 'jiyoung']
for i in fa:
    print(i, len(i))
    if len(i) > 6:
        print('긴 이름')
    elif len(i) == 6:
        print('중간 이름')
    else:
        print('짧은 이름')
    







# img = load_img('D:/과일/train/apple/apple1.jpg')

# # i = 1
# for i in range(1, 41):
#     img = load_img('D:/과일/train/apple/apple[i].jpg')
#     pd.save('D:/연습/apples[i]')
#     # i + 1
    # if i < 40:
    #     pd.save('D:\연습')
    #     continue
    # else:
    #     break

'''
img = load_img('D:/과일/train/apple/apple1.jpg', target_size=(500, 500))
x = img_to_array(img)
print(x.shape)        #(213, 237, 3)
print(x)
# x =  x.reshape(())
img.show()

'''


'''
imge_dir = 'D:/과일/train'
categories = ['apple', 'banana', 'grape', 'pineapple']
nb_classes = len(categories)

image_w = 100
image_h = 100
pixels = image_h * image_w * 3

### 이미지 파일 Data화
X = []
Y = []

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
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 700 == 0:
            print(cat, ':', f)

'''



'''


# 준비한 이미지 numpy dataset 만들기
groups_folder_path = './eyes/'
categories = ["c_eyes", "o_eyes"]

num_classes = len(categories)

print(type(categories))

x = []
y = []

for index, categorie in enumerate(categories) :
    label = [0 for i in range(num_classes)]
    label[index] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir) :
        for filename in f :
            print(image_dir + filename)
            img = cv2.imread(image_dir+filename)
            x.append(img)
            y.append(label)

x = np.array(x)
y = np.array(y)

print("x.shape :", x.shape)
print("y.shape :", y.shape)

np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)
'''