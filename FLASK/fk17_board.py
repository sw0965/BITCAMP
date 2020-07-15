from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터베이스
conn = sqlite3.connect("./DATA/wanggun.db")     # db 데이터 불러오기
cursor = conn.cursor()                          # cursor 
cursor.execute("SELECT * FROM general")         # general 데이터 다 보여주기

# print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./DATA/wanggun.db') 
    c = conn.cursor()
    c.execute("SELECT * FROM general") 
    rows = c.fetchall();
    return render_template("board_index.html", rows = rows)

    # c.close()

@app.route('/modi')
def modi():
    id = request.args.get('id')
    conn = sqlite3.connect('./DATA/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general WHERE id = '+str(id))  
    rows = c.fetchall();
    return render_template("board_modi.html", rows = rows)

@app.route('/addrec', methods=['POST', 'GET'])   # 웹 내에서 수정할수 있게 만들어주는 함수
def addrec():
    if request.method == 'POST':
        try:    # 예외처리에서 좀 더 깔끔하게 하기 위해서 try를 사용
            war = request.form['war']
            id = request.form['id']
            # conn = sqlite3.connect('./DATA/wanggun.db') 

            with sqlite3.connect("./DATA/wanggun.db") as con:
                cur = con.cursor()
                cur.execute("UPDATE general SET war = " + str(war)+ "WHERE id="+str(id))
                # cur.execute("UPDATE general SET war = " + str(war)+ "WHERE id="+str(id))
                # cur.execute("UPDATE general SET war= %s where id = %s " %(str(war), str(id1)))

                con.commit()
                msg = '정상적으로 입력되었습니다.'
                
        except:     # 에러가 떳으면 돌려놓아라
            con.rollback()
            msg = '입력과정에서 에러가 발생했습니다.'
        
        finally:    # 에러가 없으면 이게 실행됌
            con.close()

            return render_template("board_result.html", msg = msg)

app.run(host='127.0.0.1', port=5001, debug=False)





# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# # 데이터 베이스
# conn = sqlite3.connect("./DATA/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general")
# print(cursor.fetchall())


# # 웹 상에서 적용 되는 것
# @app.route('/')
# def run():
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general")
#     rows = c.fetchall();
#     return render_template("board_index.html",rows=rows)



# # 웹 상 수정
# # /modi
# @app.route('/modi')
# def modi():
#     ide = request.args.get('id')
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM general WHERE id ='+str(ide))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
#     rows = c.fetchall();
#     return render_template('board_modi.html', rows = rows)

    
    

# @app.route('/addrec',methods=['POST','GET'])
# def addrec():
#     if request.method == 'POST':
#         try:
#             conn = sqlite3.connect('./data/wanggun.db')
            
#             war = request.form['war']
#             ide = request.form['id']
#             # sqlite3.connect("./data/wanggun.db") as conn:
#             cur = conn.cursor()
#             cur.execute("UPDATE general SET war= %s WHERE id = %s " %(str(war), str(ide)))
#             # cur.execute("UPDATE general SET war = " + str(war)+ "WHERE id="+str(id))
#             conn.commit() #수정하고 나면 항상 commit
#             msg = "정상적으로 입력되었습니다."    
#         except:
#             conn.rollback()
#             msg = "입력과정에서 에러가 발생했습니다."

#         finally:
#             return render_template("board_result.html",msg = msg)
#             conn.close()
# app.run(host='127.0.0.1',port=5000,debug=False)





# 기태씨 
# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# # 데이터베이스 만들기
# conn = sqlite3.connect("./DATA/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general;")
# print(cursor.fetchall())

# @app.route('/')
# def run():
#     conn = sqlite3.connect('./DATA/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general;")
#     rows = c.fetchall()
#     return render_template("board_index.html", rows=rows)

# @app.route('/modi')
# def modi():
#     ids = request.args.get('id')
#     conn = sqlite3.connect('./DATA/wanggun.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM general where id = ' + str(ids))
#     rows = c.fetchall()
#     return render_template('board_modi.html', rows=rows)

# @app.route('/addrec', methods=['POST', 'GET'])
# def addrec():
#     if request.method == 'POST':
#         try:
#             conn = sqlite3.connect('./DATA/wanggun.db')
#             war = request.form['war']
#             ids = request.form['id']
#             c = conn.cursor()
#             # c.execute('UPDATE general SET war = ' + str(war) +  ' WHERE id = '+str(ids))
#             c.execute("UPDATE general SET war= %s WHERE id = %s " %(str(war), str(ids)))
#             conn.commit()
#             msg = '정상적으로 입력되었습니다.'
            
#         except:
#             conn.rollback()
#             msg = '에러가 발생하였습니다.'

#         finally:
#             conn.close()
#             return render_template("board_result.html", msg=msg)

# app.run(host='127.0.0.1', port=5000, debug=False)



# 쌤꺼
# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# conn = sqlite3.connect("./DATA/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general")
# print("fetchall: ",cursor.fetchall())

# @app.route("/")
# def run():
#     conn = sqlite3.connect('./DATA/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general")
#     rows = c.fetchall()
#     return render_template("board_index.html", rows = rows)

# @app.route("/modi")
# def modi():
#     id = request.args.get("id")
#     conn = sqlite3.connect('./DATA/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general where id = "+str(id))
#     rows = c.fetchall();
#     return render_template("board_modi.html", rows = rows)

# @app.route("/addred", methods = ["POST", "GET"])
# def addrec():
#     if request.method == 'POST':
#         print(request.form['war'])
#         print(request.form['id'])
#         try:
#             war = request.form['war']
#             id = request.form['id']
#             with sqlite3.connect("./DATA/wanggun.db") as con:
#                 # print("con")
#                 cur = con.cursor()
#                 cur.execute("UPDATE general SET war="+str(war)+" WHERE id="+str(id))
#                 # print("con2")
#                 con.commit()
            
#                 msg="정상적으로 입력되었습니다"

#         except:
#             con.rollback()
#             msg="입력과정에서 에러가 발생했습니다."

#         finally:
#             return render_template("board_result.html",msg = msg)
#             con.close()

# app.run(host="127.0.0.1",port=5000,debug=False)