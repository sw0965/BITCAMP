# from keras.models import load_model
# import glob, numpy as np
# from PIL import Image



# model = load_model('./mini_project/model_save/model.h5')




# # x_predict 만들기

# # from matplotlib import image
# filenames = []
# x_predict = []

# imge_dir = 'D:/과일/train/test'
# file = glob.glob(imge_dir + '/*.jpg')
# print(file)
# for i in file:
#     img = Image.open(i)
#     img = img.resize((100, 100))
#     print(img)
#     data = np.array(img)
#     print(type(data))
#     x_predict.append(data)
#     print('x_type : ', type(x_predict))  #list 형태

# x_predict = np.array(x_predict)/255
# print('type : ', type(x_predict))     # numpy 형태


# y_predict = model.predict(x_predict)

# # np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
# cnt = 0

# for i in y_predict:
#     pre_ans = i.argmax() 
    
#     # print(pre_ans)
#     pre_ans_str = ''
#     if pre_ans == 0: pre_ans_str = '( 사과 )'
#     elif pre_ans == 1: pre_ans_str = '( 바나나 )'
#     elif pre_ans == 2: pre_ans_str = '( 파인애플 )'
#     else: pre_ans_str = '( 포도 )'
#     if i[0] >= 0.5 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
#     if i[1] >= 0.5 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
#     if i[2] >= 0.5 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
#     if i[3] >= 0.5 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
#     cnt += 1