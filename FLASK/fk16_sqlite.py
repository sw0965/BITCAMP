import sqlite3

conn = sqlite3.connect("test.db")

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
                FoodName TEXT, Company TEXT, Price INTEGER)""")

# sql = "DELETE FROM supermarket"     # 테이블 안에 내용을 삭제
# '''DB이기 때문에 계속 남는다 그래서 시작할때 기존에 있던것을 삭제 해줘야됌'''
# cursor.execute(sql)


# 데이터 넣기
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \n values(?,?,?,?,?)"
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))
cursor.execute(sql, (2, '생선', '고등어', '롯데마트', 3200))
cursor.execute(sql, (3, '고기', '흑돼지', '이마트', 9800))
cursor.execute(sql, (4, '약', '박카스', '상우약국', 1200))
cursor.execute(sql, (1, '과일', '자몽', '마트', 1600))

# sql = "UPDATE supermarket SET Price = 1600 where Itemno = 1; "  # 데이터 수정해주는 코드
# cursor.execute(sql)

'''모든 작업에는 cursor.execute(설정변수) 를 해줘야 된다'''
conn.commit()

sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno, Category, FoodName, Company, Price FROM supermarket"  # 위와 같다.
cursor.execute(sql)
rows = cursor.fetchall()
print(rows)

for row in rows:
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]) + " " + str(row[4]))
    # row = cursor.fetchone()

conn.close()