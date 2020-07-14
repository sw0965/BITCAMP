import pymssql as ms
# print('잘 접속 됬나?')

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitDB')
print("끝")
cursor = conn.cursor()

cursor.execute("SELECT * FROM iris2;")

row_iris = cursor.fetchall() # 150(iris)개 중 1줄을 가져오겠다
print('DONE iris2')
cursor.execute("SELECT * FROM wine;")

row_wine = cursor.fetchall() 
print('DONE wine')

cursor.execute("SELECT * FROM sonar;")
row_sonar = cursor.fetchall() 
print('DONE sonar')

# for i in row_iris:
#     print(i)

# for i in row_wine:
#     print(i)

# for i in row_sonar:
#     print(i)


# while row:  
#     print("첫컬럼 : %s, 둘컬럼 : %s" %(row[0], row[1]))
#     row = cursor.fetchone()
# conn.commit()
# conn.close()
# cursor.execute("SELECT * FROM wine;")

# row = cursor.fetchone() # 150(iris)개 중 1줄을 가져오겠다

# while row:  
#     print("wine첫컬럼 : %s, wine둘컬럼 : %s" %(row[0], row[1]))
#     row = cursor.fetchone()

# conn.close()