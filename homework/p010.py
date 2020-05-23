# 1.3.3 연봉과 경력

''' 익명화 데이터, 연봉(salary)이 달러로, 근속 기간(tenure)이 연 단위로 표기 '''

salaries_and_tennures = [(83000, 8.7), (88000, 8.1),
                         (48000, 0.7), (76000, 6),
                         (69000, 6.5), (76000, 7.5),
                         (60000, 2.5), (83000, 10),
                         (48000, 1.9), (63000< 4.2)]

''' 더 많은 경력을 가진 사람이 더 높은 연봉을 받는다는 결과 (책 그림)
근석 연수에 따라 평균 연봉이 어떻게 달라지는지 보기'''

# 키는 근속 연수, 값은 해당 근속 연수에 대한 연봉 목록
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tennures:
    salary_by_tenure[tenure].append(salary)

# 키는 근속 연수, 값은 해당 근속 연수의 평균 연봉
average_salary_by_tenure = {
    tenure: sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}    