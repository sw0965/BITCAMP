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


def split_x(seq, size):                          
    aaa = []                                        
    for i in range(len(seq) - size + 1):            
        subset = seq[i : (i+size)]                  
        aaa.append([j for j in subset])       
                                                   
    # print(type(aaa))                                
    return np.array(aaa)   


size = 6


samsung = np.load('./data/samsung.npy', allow_pickle='ture')
hite    = np.load('./data/hite.npy', allow_pickle='ture')

print(samsung.shape) #(509, 1)
print(samsung)
print(hite.shape)    #(509, 5)

samsung = samsung.reshape(samsung.shape[0], )
print(samsung.shape) #(509,)
print(samsung)
samsung = (split_x(samsung, size))
print(samsung.shape) #(504, 6) 
print(samsung)


x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
print(x_sam)
print(x_sam.shape) 

print(y_sam) 
x_sam = x_sam.reshape(504, 5, 1)
# print(x_sam.shape) #(504, 5)
# print(y_sam.shape) #(504, )

x_hit = hite[5:510, :]
# print(x_hit.shape)   #(504, 5)


#2. 모델 구성

input1 = Input(shape=(5, 1))
x1 = LSTM(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()


#앙상블을 할땐 행까지 맞춰줘야한다.

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')
model.fit([x_sam, x_hit], y_sam, epochs=5)
