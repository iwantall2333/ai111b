# 100%原創
import random
import copy

def total(point,s): # 算總距離
    sum1 = 0

    pointTM = point.copy()
    length = len(point)
    for i in range (length): # 0~ length-1
        pointTM[i] = point[s[i]]  # 變成換過順序的point

    for i in range(length-1):
        sum1 += (abs(pointTM[i][0] -pointTM[i+1][0]) + abs(pointTM[i][1] -pointTM[i+1][1])) ** 0.5

    sum1 += (abs(pointTM[length-1][0] -pointTM[0][0]) + abs(pointTM[length-1][1] -pointTM[0][1])) ** 0.5

    return sum1


def start():

    #setting
    s = [0,1,2,3,4,5,6,7,8,9] 
    point=[[0,0],[1,0],[2,0],[3,0],
            [0,1],[0,2],[0,3],
            [3,1],[3,2],[3,3]]

    random.shuffle(s)
    print("initial",s)
    print("initial length: ",total(point,s))
    
    temp=total(point,s)

    # code
    for i in range(10000):
        rd1 = random.randint(0,9)
        rd2 = random.randint(0,9)
        s[rd1],s[rd2]=s[rd2],s[rd1]
        if(total(point,s)>temp):
            s[rd1],s[rd2]=s[rd2],s[rd1] # 沒超過就換回來
        else:
            temp=total(point,s)

    print("length: ",total(point,s)," set: ",s)
   

start()

'''
執行結果
ai111b>python 習題2.py
initial [6, 3, 4, 5, 1, 9, 2, 8, 7, 0]
initial length:  17.8817101429896
length:  10.732050807568877  set:  [6, 5, 4, 0, 1, 2, 3, 7, 8, 9]
'''

