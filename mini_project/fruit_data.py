from PIL import Image
import glob

img_path1 = 'D:/과일/test/apple/apple1.jpg'

im1 = Image.open(img_path1)
print('{}'.format(im1.format))
print('size: {} '.format(im1.size))
print('image mode: {} '.format(im1.mode))
# im1.show()

image_list1 = []
resized_image1 = []

for filename in glob.glob('D:/과일/test/apple/*.jpg'):
    print(filename)
    img = Image.open(filename)
    image_list1.append(img)


img_path2 = 'D:/과일/test/banana/banana1.jpg'

im2 = Image.open(img_path2)
print('{}'.format(im2.format))
print('size: {} '.format(im2.size))
print('image mode: {} '.format(im2.mode))

image_list2 = []
resized_image2 = []

for filename in glob.glob('D:/과일/test/banana/*.jpg'):
    print(filename)
    img2 = Image.open(filename)
    image_list2.append(img)