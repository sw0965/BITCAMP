import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA



# def split_x(seq, size):
#     aaa = []
#     for i in range(len(seq)):
#         subset = seq[i:(i+size)]
#         aaa.append([j for j in subset])
#     return np.array(aaa)

#split 변수 지정
def split_x(seq, size):                          
    aaa = []                                        
    for i in range(len(seq) - size + 1):            
        subset = seq[i : (i+size)]                  
        aaa.append([j for j in subset])       
                                                   
    # print(type(aaa))                                
    return np.array(aaa)   
size = 6


#데이터 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle='ture')
hite    = np.load('./data/hite.npy', allow_pickle='ture')
print(samsung.shape) #(509, 1)
print(samsung)
print(hite.shape)    #(509, 5)

#samsung 스케일링
scaler = StandardScaler()
scaler.fit(samsung)   
st_sam = scaler.transform(samsung)
print('st_sam :', st_sam)
print('st_sam.shape :', st_sam.shape) #(509, 1)

# #samsung pca
# pca = PCA(n_components=1)
# pca.fit(st_sam)
# pca_sam = pca.transform(st_sam)
# print('pca_sam :', pca_sam)
# print('pca_sam.shape :', pca_sam.shape) #(509, 1)

#y 값을 주기위해 뒤에를 nan으로 만들기(어차피 509, 1이라 가능한것)
re_sam = st_sam.reshape(st_sam.shape[0], )
print('re_sam.shape :', re_sam.shape) #(509,)

#x,y로 나누기 위해 y를 6으로 만들어서 (504,5) (504,1) 로 자르기위해 스플릿
sp_sam = (split_x(re_sam, size))
print('sp_sam : ', sp_sam.shape) #(504, 6) 

#(504,5) (504,1) 이렇게 슬라이싱
x_sam = sp_sam[:, 0:5]
y_sam = sp_sam[:, 5]
print(sp_sam.shape) #(504, 5)
print(sp_sam.shape) #(504,)

#lstm으로 연동하기 위해 차원변경
x_sam = x_sam.reshape(504, 5, 1)
print(x_sam.shape) #(504, 5, 1)




########################################

# hite 스케일링
scaler = StandardScaler()
scaler.fit(hite)   
st_hite = scaler.transform(hite)
print('st_hite :', st_hite)
print('st_hite.shape :', st_hite.shape) #(509, 5)

# hite pca로 차원 축소
pca = PCA(n_components=1)
pca.fit(st_hite)
pca_hite = pca.transform(st_hite)
print('pca_hite :', pca_hite)
print('pca_hite.shape :', pca_hite.shape) #(509, 1)

#hite 스케일 뒤 스플릿
sp_hite = (split_x(pca_hite, size))
print('sp_hite :', sp_hite)
print('sp_hite.shape :', sp_hite.shape) #(504, 6, 1)





#2. 모델 구성

input1 = Input(shape=(5, 1))
x1 = LSTM(10)(input1)
x1 = Dense(20)(x1)
x1 = Dense(40)(x1)
x1 = Dense(80)(x1)
x1 = Dense(40)(x1)

input2 = Input(shape=(6, 1))
x2 = LSTM(5)(input2)
x2 = Dense(10)(x2)
x2 = Dense(50)(x2)
x2 = Dense(100)(x2)
x2 = Dense(50)(x2)

# merge = concatenate([x1, x2])
merge = Concatenate()([x1, x2])


output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()


#앙상블을 할땐 행까지 맞춰줘야한다.

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit([x_sam, sp_hite], y_sam, epochs=1)

loss,mse = model.evaluate([x_sam, sp_hite], y_sam)
print('loss :', loss)
print('mse :', mse)

# y_predict = model.predict()