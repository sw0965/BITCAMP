import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


###########################################################################
###########################################################################
################   hite 를 전처리 후 슬라이싱 pca 적용 x    ################
###########################################################################
###########################################################################



######################################   split 변수 지정 ####################################


def split_x(seq, size):                          
    aaa = []                                        
    for i in range(len(seq) - size + 1):            
        subset = seq[i : (i+size)]                  
        aaa.append([j for j in subset])                                                          
    return np.array(aaa)   
size = 6


##################################  데이터 불러오기  ######################################


samsung = np.load('./data/samsung.npy', allow_pickle='ture')
hite    = np.load('./data/hite.npy', allow_pickle='ture')
print(samsung.shape) #(509, 1)
print(hite.shape)    #(509, 5)


############################  hite를 train,test 하기위해 슬라이싱  ############################


sli_hite = hite[5:510, :4]
print(sli_hite.shape) #(504, 4)


#####################################  sam 전처리  ###########################################


#y 값을 주기위해 뒤에를 nan으로 만들기(어차피 509, 1이라 가능한것)
re_sam = samsung.reshape(samsung.shape[0], )
print('re_sam.shape :', re_sam.shape) #(509,)

#x,y로 나누기 위해 y를 6으로 만들어서 (504,5) (504,1) 로 자르기위해 스플릿
sp_sam = (split_x(re_sam, size))
print('sp_sam : ', sp_sam.shape) #(504, 6) 

#(504,5) (504,1) 이렇게 슬라이싱
x_sam = sp_sam[:, 0:5]
y_sam = sp_sam[:, 5]


################################    train test 분류     ####################################


x1_train_sam, x1_test_sam, x2_train_hite, x2_test_hite, y1_train_sam, y1_test_sam = train_test_split(
    x_sam, sli_hite, y_sam, shuffle=False, train_size=0.8)
print('x1_train_sam.shape : ', x1_train_sam.shape)   #(403, 5)
print('x1_test_sam.shape : ', x1_test_sam.shape)     #(101, 5)
print('x2_train_hite.shape : ', x2_train_hite.shape) #(403, 4) 
print('x2_test_hite.shape : ', x2_test_hite.shape)   #(101, 4)
print('y1_train_sam.shape : ', y1_train_sam.shape)   #(403, )
print('y1_test_sam.shape : ', y1_test_sam.shape)     #(101, )


################################   samsung  train 스케일링   ######################################


scaler = StandardScaler()
scaler.fit(x1_train_sam)   
st_train_sam = scaler.transform(x1_train_sam)
print('st_sam :', st_train_sam)
print('st_sam.shape :', st_train_sam.shape) #(403, 5)


#################     lstm으로 연동하기 위해 Samsung train 데이터 차원변경     ##################


x1_train_sam = st_train_sam.reshape(403, 5, 1)
print(x1_train_sam.shape) #(403, 5, 1)


###################################   hite train 스케일링   ##################################

 
scaler = StandardScaler()
scaler.fit(x2_train_hite)   
st_train_hite = scaler.transform(x2_train_hite)
print('st_hite :', st_train_hite)
print('st_hite.shape :', st_train_hite.shape) #(403, 4)


#####################  lstm으로 연동하기 위해 hite train 데이터 차원변경  ######################


x2_train_hite = st_train_hite.reshape(403, 4, 1)
print(x2_train_hite.shape) #(403, 4, 1)


################################################################################################
'''
# # hite pca로 차원 축소
# pca = PCA(n_components=1)
# pca.fit(st_hite)
# pca_hite = pca.transform(st_hite)
# print('pca_hite :', pca_hite)
# print('pca_hite.shape :', pca_hite.shape) #(509, 1)
'''



'''
x2_train_hite = (split_x(st_train_hite, size))
print('sp_hite :', x2_train_hite)
print('sp_hite.shape :', x2_train_hite.shape) #(504, 6, 1)
'''
################################        test값 reshape       ###################################


x1_test_sam  = x1_test_sam.reshape(101, 5, 1)
x2_test_hite = x2_test_hite.reshape (101, 4, 1)


######################################    모델 구성     ########################################


input1 = Input(shape=(5, 1))
x1     = LSTM(5)(input1)
x1     = Dense(10)(x1)
x1     = Dense(20)(x1)
x1     = Dense(40)(x1)
x1     = Dense(120)(x1)
x1     = Dense(40)(x1)
x1     = Dense(20)(x1)
x1     = Dense(10)(x1)
x1     = Dense(5)(x1)

input2 = Input(shape=(4, 1))
x2     = LSTM(4)(input2)
x2     = Dense(8)(x2)
x2     = Dense(32)(x2)
x2     = Dense(128)(x2)
x2     = Dense(376)(x2)
x2     = Dense(128)(x2)
x2     = Dense(32)(x2)
x2     = Dense(8)(x2)
x2     = Dense(4)(x2)

merge = concatenate([x1, x2])
# merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()


########################################    훈련    ############################################


model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit([x1_train_sam, x2_train_hite], y1_train_sam, epochs=50)


########################################    평가    ############################################


loss,mse = model.evaluate([x1_test_sam, x2_test_hite], y1_test_sam)
print('loss :', loss)
print('mse :', mse)


########################################    예측    ############################################


y_predict = model.predict([x1_test_sam, x2_test_hite])
print(y_predict)


###############################################################################################
# 슬라이싱할때 일자가 몇개 짤려서 서로 일자가 다름 내일 맞춰주기.