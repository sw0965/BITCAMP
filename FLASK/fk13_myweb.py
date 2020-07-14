import pyodbc as pyo

server = '127.0.0.1'
database = 'bitDB'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; Server=' +server+'; PORT=1433; DATABASE=' +database+'; UID=' +username+'; PWD=' +password)

curser = conn.cursor()

tsql = 'SELECT * FROM iris2;'

# with curser.execute(tsql):
#     row = curser.fetchone()

#     while row :
#         print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]) + " " + str(row[4]))
#         row = curser.fetchone()

from flask import Flask, render_template
app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    curser.execute(tsql)    # iris2를 웹에서 실행하겠다.
    return render_template("myweb.html", rows = curser.fetchall())

if __name__ == '__main__':
    app.run(host='127.0.0.10', port = 8800, debug=False)

conn.close()