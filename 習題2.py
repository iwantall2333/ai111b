# 100%原創
import random
import copy

def total(point,s): # 算總距離
    sum1 = 0

    pointTM = point.copy()
    length = len(point) -1
    for i in range (length+1): # 0~ length-1
        pointTM[i] = point[s[i]]  # 變成換過順序的point

    for i in range(length):
        sum1 += (abs(pointTM[i][0] -pointTM[i+1][0]) + abs(pointTM[i][1] -pointTM[i+1][1])) ** 0.5

    sum1 += (abs(pointTM[length][0] -pointTM[0][0]) + abs(pointTM[length][1] -pointTM[0][1])) ** 0.5

    return sum1


def start():

    #setting
    s = [0,1,2,3,4,5,6,7,8,9] 
    point=[[0,0],[1,0],[2,0],[3,0],
            [0,1],[0,2],[0,3],
            [3,1],[3,2],[3,3]]

    random.shuffle(s)
    print(s)
    rd1 = random.randint(0,9)
    a = s[rd1]
    rd2 = random.randint(0,9)
    b= s[rd2]
    temp=len(point)*10+1

    # code
    for i in range(100000):
        if(total(point,s)>temp):
            s[rd1],s[rd2]=s[rd2],s[rd1] # 沒超過就交換

    print("length: ",total(point,s)," set: ",s)
   

start()

'''
執行結果
ai111b>python 習題2.py
[8, 0, 4, 3, 1, 9, 2, 7, 6, 5]
length:  16.26868186481444  set:  [8, 0, 4, 3, 1, 9, 2, 7, 6, 5]

ai111b>python 習題2.py
[5, 6, 0, 2, 7, 9, 4, 3, 1, 8]
length:  14.861041012060838  set:  [5, 6, 0, 2, 7, 9, 4, 3, 1, 8]

ai111b>python 習題2.py
[7, 8, 6, 5, 0, 9, 1, 4, 2, 3]
length:  15.246035652598035  set:  [7, 8, 6, 5, 0, 9, 1, 4, 2, 3]
'''

