import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/study/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

warnings.filterwarnings('ignore')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)


warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')  # 모든 클래스파이어값들이 들어있음

for (name, algorithm) in allAlgorithms:   # 올알고리즘에서 반환하는 값이 (네임, 알고리즘)
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 = ', accuracy_score(y_test, y_pred))
'''
AdaBoostClassifier 의 정답률 =  0.9666666666666667
BaggingClassifier 의 정답률 =  0.9666666666666667
BernoulliNB 의 정답률 =  0.3
CalibratedClassifierCV 의 정답률 =  0.9666666666666667
ComplementNB 의 정답률 =  0.7
DecisionTreeClassifier 의 정답률 =  0.9666666666666667
ExtraTreeClassifier 의 정답률 =  0.9
ExtraTreesClassifier 의 정답률 =  0.9333333333333333
GaussianNB 의 정답률 =  0.9333333333333333
GaussianProcessClassifier 의 정답률 =  0.9666666666666667
GradientBoostingClassifier 의 정답률 =  0.9333333333333333
KNeighborsClassifier 의 정답률 =  0.9666666666666667
LabelPropagation 의 정답률 =  0.9666666666666667
LabelSpreading 의 정답률 =  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 =  1.0
LinearSVC 의 정답률 =  0.9666666666666667
LogisticRegression 의 정답률 =  1.0
LogisticRegressionCV 의 정답률 =  0.9
MLPClassifier 의 정답률 =  1.0
MultinomialNB 의 정답률 =  0.8666666666666667
NearestCentroid 의 정답률 =  0.9
NuSVC 의 정답률 =  0.9666666666666667
PassiveAggressiveClassifier 의 정답률 =  0.7
Perceptron 의 정답률 =  0.5333333333333333
QuadraticDiscriminantAnalysis 의 정답률 =  1.0
RadiusNeighborsClassifier 의 정답률 =  0.9333333333333333
RandomForestClassifier 의 정답률 =  0.9333333333333333
RidgeClassifier 의 정답률 =  0.8333333333333334
RidgeClassifierCV 의 정답률 =  0.8333333333333334
SGDClassifier 의 정답률 =  0.7
SVC 의 정답률 =  0.9666666666666667
'''
# sklearn 0.22.1 버전에선 안돌아감 새로운게 들어오고 안쓰는거 나가다보니 그래서 0.20.1 로변경
# 모델들 31개
import sklearn
print(sklearn.__version__)
