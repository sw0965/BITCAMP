from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 데이터
dataset = load_boston()
x       = dataset.data
y       = dataset.target


# 전처리
scaler = StandardScaler()
scaler.fit(x)   
x = scaler.transform(x)

# 트레인 테스트 스플릿
x_train,x_test,y_train,y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

# 전처리
scaler = StandardScaler()
scaler.fit(x)   
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)



# 모델    
model = SVC()                    
# model = LinearSVC()             
# model = KNeighborsClassifier()   
# model = KNeighborsRegressor()    
# model = RandomForestClassifier()  
# model = RandomForestRegressor()   


# 훈련 
model.fit(x_train, y_train)
score = model.score(x_test, y_test) #evaluate 같은거
print('score = ', score)
y_predict = model.predict(x_test)
# print(y_predict)
acc = accuracy_score(y_test, y_predict)  
print("acc = ", acc)
r2 = r2_score(y_test,y_predict)
print("r2 = ", r2)


