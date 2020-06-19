#미니 프로젝트 

cnt =[0,0,0,0]
names=["사과","바나나","포도","파인애플"]
prices=[1200,1800,2900,3800]

for i in y_predict:
    if i == 0:
        cnt[0] +=1
    elif i ==1:
        cnt[1] +=1
    elif i ==2:
        cnt[2] +=1
    elif i ==3:
        cnt[3] +=1


for i in range(4):
    print(names[i], str(cnt[i])+'개 입니다.'' 개당 가격은 ',str(prices[i])+'원이며, 총 가격은 ',str(cnt[i]*prices[i])+'원 입니다.',sep="")
