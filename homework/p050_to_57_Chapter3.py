### 선그래프 ###
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# x 축에 연도, y축에 gdp가 있는 선 그래프 만들기
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# 제목 정하기
plt.title("Nominal GDP")

# y축에 레이블을 추가하기
plt.ylabel("Billions of $")
plt.show()

#### 막대 그래프 ####
movies = ["Annie Hall", "Ben-Hur", "Cassblanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 0,1,2,3,4 y 좌표는 num_oscars 로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My favourite Movies") # 제목을 추가
plt.ylabel(" # Of Academy Awards") # Y축에 레이블을 추가

# x 축 각 막대의 중앙에 영화 제목을 레이블로 추가
plt.xticks(range(len(movies)), movies)

plt.show()

###히스토그램###
from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화 한다. 100점은 90점대에 속한다sss
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],   # 각 막대를 오른쪽으로 5만큼 옮기고
         histogram.values(),                 # 각 막대의 높이를 정해 주고
         10,                                 # 너비는 10으로 하고
         edgecolor = (0, 0, 0))              # 각 막대의 테두리는 검은색으로 설정하자

plt.axis([-5, 105, 0, 5])                    # x축은 -5 부터 105
                                             # y축은 0부터 5

plt.xticks([10 * i for i in range(11)])      # x 축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()             
######################################################################
mentions = [500, 505]                                                #
years = [2017, 2018]                                                 #
                                                                     #
plt.bar(years, mentions, 0.8)                                        #
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# 이렇게 하지 않으면 matplotlib이 x 축에 0,1 레이블을 달고
# 주변부 어딘가에 =0.013e3이라고 표기해 둘 것이다. 
plt.ticklabel_format(useOffset=False)

# 오해를 불러일으키는 y축은 500 이상의 부분만 보여 줄 것이다.
plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")
plt.show()

########################
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
plt.show()
####################################################################

# 선 그래프
variance = [1,2,4,8,16,32,64,128,256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# 한 차트에 여러 개의 선을 그리기 위해
# plt.plot 을 여러 번 호출할 수 있다.
plt.plot(xs, variance, 'g-', label='variance')       # 실선
plt.plot(xs, bias_squared, 'r-', label='bias^2')     #일점쇄선
plt.plot(xs, total_error, 'b:', label='total error') #점선

# 각 선에 레이블을 미리 달아 놨기 때문에
# 범례(legend)를 쉽게 그릴 수 있다. 
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()

# 산점도

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 150, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends,minutes)

# 각 포인트에 레이블을 달자.
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
    xy=(friend_count, minute_count),   # 레이블을 데이터 포인트 근처에 두되
    xytext=(5, -5),                    # 약간 떨어져 있게 하자.
    textcoords='offset points')

plt.title("Daily Minutes vs. Number od Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.show()



