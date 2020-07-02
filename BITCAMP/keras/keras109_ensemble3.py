import numpy as np
x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2. 모델구성 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(1,))
x2 = Dense(100)(input2)
x2 = Dense(100)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

x3 = Dense(100)(merge)
output1 = Dense(1)(x3)



x4 = Dense(70)(merge)
x4 = Dense(70)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs= [input1,input2], outputs=[output1, output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss=['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse','acc'], loss_weights=[0.1, 0.9])    # loss weight=데이터가 두개라 가능한 파라미터 분류모델에 가중치를90퍼 주겠다.
model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate([x1_train,x2_train], [y1_train, y2_train])
print('loss: ', loss)

x1_pred = np.array([11, 12, 13, 14])
x2_pred = np.array([11, 12, 13, 14])

y_pred = model.predict([x1_pred,x2_pred])
print(y_pred)

'''
loss weight 안썻을때

loss:  [0.6818706393241882, 1.4065716641198378e-05, 0.6818565726280212, 1.4065716641198378e-05, 1.0, 0.24446193873882294, 0.6000000238418579]
[array([[11.004395],
       [12.00553 ],
       [13.006665],
       [14.007801]], dtype=float32), array([[0.34401727],
       [0.3240956 ],
       [0.30479148],
       [0.28615043]], dtype=float32)]
'''

'''
결론 소용이 없었다.

loss:  [0.6155335903167725, 0.015641041100025177, 0.6821883320808411, 0.015641041100025177, 1.0, 0.24456055462360382, 0.5]
[array([[11.265698],
       [12.300004],
       [13.334311],
       [14.368614]], dtype=float32), array([[0.42879128],
       [0.41450423],
       [0.40035957],
       [0.3863791 ]], dtype=float32)]
'''