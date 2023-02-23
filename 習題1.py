import matplotlib.pyplot as plt
import numpy as np

# x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
# y = np.array([2, 3, 4, 5, 6], dtype=np.float32)
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)# 原始x y座標，要找到他們的回歸線

def predict(a1,a2, xt):
	return a1+a2*xt		# y = a0 + a1*x

def MSE(a1,a2, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a1,a2,x[i]))**2
	return total

def loss(a1,a2):
	return MSE(a1,a2, x, y)

# p = [0.0, 0.0]
# plearn = optimize(loss, p, max_loops=3000, dump_period=1)
def optimize():
    # predict 爬山a0 a1的值，套入x[]算出y，再MSE，一直找MSE的最小
	p=[0,0]
	h=0.001
	while(True):
		if(loss(p[0]+h,p[1])<loss(p[0],p[1])):
			p[0] = p[0]+h
		elif(loss(p[0]-h,p[1])<loss(p[0],p[1])):
			p[0] = p[0]-h
		elif(loss(p[0],p[1]+h)<loss(p[0],p[1])):
			p[1] = p[1]+h
		elif(loss(p[0],p[1]-h)<loss(p[0],p[1])):
			p[1] = p[1]-h
		else:
			break
	return p
    # p = [2,1] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線) # 找x,y值; p是a0 a1兩個參數
    # p = [3,2] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
   

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
