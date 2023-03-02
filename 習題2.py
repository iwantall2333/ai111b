import random
import copy

def total(point): # 算總距離
	sum1 = 0
	for i in range(len(point)-1):
		sum1 += (abs(point[i][0]-point[i+1][0])+abs(point[i][1]-point[i+1][1])) ** 0.5
	return sum1

def meter(a,b,point): # 暫時算總距離

    sum = 0
    pointTM = point.copy()
    pointTM[a],pointTM[b] = pointTM[b],pointTM[a] 

    for i in range(len(point)-1):
        sum += (abs(pointTM[i][0] -pointTM[i+1][0]) + abs(pointTM[i][1] -pointTM[i+1][1])) ** 0.5
    return sum

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
        if(meter(a,b,point)<temp):
            s[rd1],s[rd2]=s[rd2],s[rd1]

    print("length: ",total(point)," set: ",s)
   

start()

'''
執行結果
ai111b>python 習題2.py
[6, 5, 9, 2, 0, 1, 3, 4, 7, 8]
length:  11.23606797749979  set:  [6, 5, 9, 2, 0, 1, 3, 4, 7, 8]

ai111b>python 習題2.py
[3, 8, 4, 9, 0, 7, 1, 2, 6, 5]
length:  11.23606797749979  set:  [3, 8, 4, 9, 0, 7, 1, 2, 6, 5]

ai111b>python 習題2.py
[6, 9, 8, 1, 2, 7, 5, 3, 4, 0]
length:  11.23606797749979  set:  [6, 9, 8, 1, 2, 7, 5, 3, 4, 0]
'''

