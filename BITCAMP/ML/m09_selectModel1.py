import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators
import warnings
from   sklearn.datasets         import load_boston

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/study/boston_house_prices.csv', header=1)

print(boston.shape)
print(boston)


dataset = load_boston()
x       = dataset.data
y       = dataset.target

# x = boston.iloc[:, 0:13]
# y = boston.iloc[:, 13]

# print(x)
# print(y)

warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='regressor')  # 모든 클래스파이어값들이 들어있음

for (name, algorithm) in allAlgorithms:   # 올알고리즘에서 반환하는 값이 (네임, 알고리즘)
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 = ', r2_score(y_test, y_pred))
'''

'''
# sklearn 0.22.1 버전에선 안돌아감 새로운게 들어오고 안쓰는거 나가다보니 그래서 0.20.1 로변경
# 모델들 31개
import sklearn
print(sklearn.__version__)
