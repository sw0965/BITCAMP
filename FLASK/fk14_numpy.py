import pymssql as ms
import numpy as np

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitDB')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
print(row)
conn.close()

print("_=_=_=_=_=_=_=_=_=_=_=_=_=_=_")
aaa= np.asarray(row)
print(aaa)
print(aaa.shape)    # (150, 5)

np.save('./DATA/test_flask_iris2.npy', aaa)