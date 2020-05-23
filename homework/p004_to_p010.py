# 1.3.1 핵심 인물 찾기

users = [{"id": 0, "name": "Hero" },
         {"id": 1, "name": "Dunn" },
         {"id": 2, "name": "Sue" },
         {"id": 3, "name": "Chi" },
         {"id": 4, "name": "Thor" },
         {"id": 5, "name": "Clive" },
         {"id": 6, "name": "Hicks" },
         {"id": 7, "name": "Devin" },
         {"id": 8, "name": "Kate" }, 
         {"id": 9, "name": "Klein" }]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#  사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
friendships = {user["id"]: [] for user in users}

'''
친구 관계를 쌍의 리스트로 표현하는 것이 이것을 다루는 가장 쉬운 방법은 아니다. 
가령, id 1인 사용자의 모든 친구를 찾으려면 모든 쌍을 순회하여 1이 포함되어 있는 쌍을 구해야 한다.
만약 엄청나게 많은 쌍이 주어졌다면 특정 사용자의 모든 친구를 찾기 위해 굉장히 오랜 시간이 걸릴 것이다. 
대신,사용자 id를 키(key)로 사용하고 해당 사용자의 모든 친구 목록을 값(value)으로 구성한 딕셔너리를 생성해 보자. 
(딕셔너리를 통한 데이터 탐색은 매우 빠르다)
'''
# friendship_pairs 내 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

'''
이렇게 각 사용자의 친구 목록을 딕셔너리로 만들면 
'네트워크상에서 각 사용자의 평균 연결 수는 몇 개인가?' 와 같이 네트워크의 특성에 관한 질문에 답할 수 있다.
이 질문에 답하기 위해 먼저 friendships 안 모든 리스트의 길이를 더해서 총 연결 수를 구해 보자.
'''
def number_of_friends(user):
    """user의 친구는 몇 명일까?"""
    user_id = user["id"]    
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user)
                        for user in users)       # 24
'''
이제 단순히 이 합을 사용자의 수로 나누면 된다.
'''
num_user = len(users)                            # 총 사용자 리스트의 길이
avg_connection = total_connections / num_user    # 24 / 10 == 2.4
'''
다음으로 연결 수가 가장 많은 사람, 즉 친구가 가장 많은 사람이 누군지 알아 보자.
사용자의 수가 많지 않으므로 '친구가 제일 많은 사람' 부터 '제일 적은 사람' 순으로 사용자를 정렬해 보자.
'''
# (user_id, number_of_friends)로 구성된 리스트 생성
num_friends_by_id = [(user["id"], number_of_friends(user))
                      for user in users]

num_friends_by_id.sort(                          # 정렬해 보자.
    key=lambda id_and_friends: id_and_friends[1], # num_friends 기준으로
    reverse=True)                                 # 제일 큰 숫자부터 제일 작은 숫자순으로

# (user_id, num_friends) 쌍으로 구성되어 있다.
# [(1, 3),(2, 3),(3, 3),(5, 3),(8, 3),
#  (0, 2),(4, 2),(6, 2),(7, 2),(9, 1)]
'''
친구 추천 기능 설계
친구의 친구 소개: 각 사용자의 친구에 대해 그 친구의 친구들을 살펴보고, 사용자의 모든 친구에 대해 똑같은 작업을 반복하고 결과를 저장.
'''

# 1.3.2 데이터 과학자 추천하기

def foat_ids_bad(user):
    #"foaf"는 친구의 친구("friend of a friend") 를 의미하는 약자다.
    return [foaf_id
            for friend_id in friendships[user["id"]]
            for foaf_id in friendships[friend_id]]

# 이 함수를 user[0], 즉 Hero에 관해 실행하면 다음 결과가 반환된다.
[0, 2, 3, 0, 1, 3] # Hero도 자신의 친구의 친구이므로, 사용자 0(자기 자신)이 두번 포함되어 있다.
                   # 그리고 이미 Hero와 친구인 사용자 1과 사용자 2도 포함되어 있는 것을 볼 수 있다.
                   # 사용자 3인 Chi는 두 명의 친구와 친구이기 때문에 두 번 포함되어 있다.

print(friendships[0])   # [1, 2]
print(friendships[1])   # [0, 2, 3]                
print(friendships[2])   # [0, 1, 3]                

'''
서로가 아는 친구가 몇명인지
동시에 사용자가 이미 아는 사람을 제외하는 함수 만들기.
'''
from collections import Counter                  # 별도로 import해 주어야 한다.

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]    # 사용자의 친구 개개인에 대해
        for foaf_id in friendships[friend_id]    # 그들의 친구들을 세dj 보고
        if foaf_id != user_id                    # 사용자 자신과
        and foaf_id not in friendships[user_id]) # 사용자의 친구는 제외

print(friends_of_friends(users[3]))              # Counter({0: 2, 5: 1}) Chi(id:3)는 Hero(id:0)와 함께 아는 친구가 두 명이고, Clive와 함께 아는 친구가 1명

'''
관심사 엮기 (user_id, interest)
'''
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBbase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "refression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "Programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")]

'''
특정 관심사를 공유하는 사용자들을 찾아 주는 함수 만들기
'''
def data_scientists_who_like(target_interest):
    '''특정 관심사를 갖고 있는 모든 사용자 id를 반환해 보자.'''
    return [user_id
    for user_id, user_interest in interests
    if user_interest == target_interest]

'''
그러나 이 코드는 호출할 때마다 관심사 덷이터를 매번 처음부터 끝까지 훑어야 한다는 단점이 있다. 
사용자 수가 많고 그들의 관심사가 많다면(또는 데이터를 여러 번 훑을 거라면)
각 관심사로 사용자 인덱스(index)를 만드는 것이 나을지 모른다.
'''

from collections import defaultdict

# 키가 관심사, 값이 사용자 id 
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

''' 더불어 각 사용자에 관한 관심사 인덱스도 만들기'''

# 키가 사용자 id, 값이 관심사
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

'''이제 특정 사용자가 주어졌을 때, 사용자와 가장 유사한 관심사를 가진 사람이 누구인지 다음의 3단계로 알 수 있다.
- 해당 사용자의 관심사들을 훑는다.
- 각 관심사를 가진 다른 사용자들이 누구인지 찾아 본다.
- 다른 사용자들이 몇 번이나 등장하는지 센다.

위의 과정들을 다음과 같은 코드로 구현할 수 있다.'''

def most_common_interests_with(user):
    return Counter(
        interests_by_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"])
        