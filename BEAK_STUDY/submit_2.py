# https://www.acmicpc.net/problem/10039
#------------평균 점수 문제 (브4)-------------#
원섭 = int(input())
세희 = int(input())
상근 = int(input())
숭   = int(input())
강수 = int(input())

student_score = [원섭, 세희, 상근, 숭, 강수]
study = []
for i in student_score:
    # print(average)
    if i < 40:
        study.append(40)
    else:
        study.append(i)
average = sum(study)//len(study)
print(average)

#소수점이 나와서 풀리지 틀렸습니다가 떴다.