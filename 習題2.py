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
    print(s)
    
    temp=len(point)*10+1

    # code
    for i in range(100000):
        rd1 = random.randint(0,9)
        rd2 = random.randint(0,9)
        s[rd1],s[rd2]=s[rd2],s[rd1]
        if(total(point,s)>temp):
            s[rd1],s[rd2]=s[rd2],s[rd1] # 沒超過就換回來

    print("length: ",total(point,s)," set: ",s)
   

start()

'''
執行結果
ai111b>python 習題2.py      
[3, 8, 9, 5, 1, 6, 7, 0, 4, 2]
length:  16.11438315501064  set:  [3, 8, 9, 5, 1, 6, 7, 0, 4, 2]

ai111b>python 習題2.py
[6, 2, 8, 1, 5, 7, 4, 9, 3, 0]
length:  18.864440800412844  set:  [6, 2, 8, 1, 5, 7, 4, 9, 3, 0]

ai111b>python 習題2.py
[1, 8, 6, 5, 3, 7, 4, 9, 0, 2]
length:  17.06789006772473  set:  [1, 8, 6, 5, 3, 7, 4, 9, 0, 2]
'''

