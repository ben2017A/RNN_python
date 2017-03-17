import numpy as np
import math
#用RNN实现sale时间序列的预测
import csv

data = csv.reader(open('data1.csv', encoding='utf-8'))
num=0
X=np.array([float(row[1]) for row in data])
print(X)
for row in X:
	num=num+1
#i:1099 ridge:769
print(num)
ridge=int(num/10*7)
print(num,ridge)
i=0
window=10 #窗口长度是指用前10天预测后1天的数据
#所以窗口长度就是T，会分为多个小时间序列
size1=ridge-window+1
X_train=np.ones((size1,window))
Y_train=np.ones((size1,1))
while i<size1:
	X_train[i,:]=X[i:i+window]
	Y_train[i,:]=X[i+window+1]
	i=i+1
print("i ",i)
size2=num-size1
X_test=np.ones((size2,window))
Y_test=np.ones((size2,1))
while i<num-window-1:
	X_test[i-size1,:]=X[i:i+window]
	Y_test[i-size1,:]=X[i+window+1]
	i=i+1
#输入层有1个节点
#隐层自己决定4个节点？
#输出层有1个节点
#st=tanh(U*xt+W*st-1) ot=V*st

#首先明确的是self只有在类的方法中才会有，独立的函数或方法是不必带有self的。self在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
#self 指向类的实例对象
def sigmoid(x):
	return 1/(1+math.exp(-x))
class RNNNumpy:
	def __init__(self,in_dim,hidden_dim=4,out_dim=1,bptt_truncate=4):
		print("init")
		self.in_dim=in_dim
		self.hidden_dim=hidden_dim
		self.out_dim=out_dim
		#random.rand:返回随机数，random.randn:返回满足标准正态分布的样本
		self.U=np.random.randn(in_dim,hidden_dim)*np.sqrt(2.0/hidden_dim)#输入层->隐层
		self.V=np.random.randn(hidden_dim,out_dim)*np.sqrt(2.0/out_dim)#隐层->输出层
		self.W=np.random.randn(hidden_dim,hidden_dim)*np.sqrt(2.0/hidden_dim)#隐层->隐层
		self.bh=np.zeros((1,hidden_dim))#隐层
		self.bk=np.zeros((1,out_dim))#输出层
		self.seta=np.zeros((1,hidden_dim))
	#x是指一个小的时间序列，即10个数
	def forward_pass(self,x):
		print("forward_pass")
		#T是time steps
		T=len(x)
		#先计算某一个时刻xt的隐层输出st以及输出层输出ot
		s=np.zeros((T+1,self.hidden_dim))
		s[-1]=np.zeros((1,self.hidden_dim))#因为时刻是0开始，但是为了不用特殊化，0时刻仍然有隐层的输入[0,0,0......0]不过为什么是T+1？？？
		o=np.zeros((T,self.out_dim))
		#print("UVW---------foward")
		#print(self.U,self.V,self.W)
		#for each time step 隐层激活函数用tanh，输出层是对隐层输出的线性组合
		for t in np.arange(T):
			#print(t)
			#s ,o 分别为隐层，输出层的输出
			#print("t:输入",t," ",x[t]," ",np.dot(x[t],self.U)+np.dot(s[t-1],self.W))
			s[t]=np.tanh(np.dot(x[t],self.U)+np.dot(s[t-1],self.W)+self.bh-self.seta)
			o[t]=np.dot(s[t],self.V)+self.bk
		return [s,o]
	#RNNNumpy.forward_pass=forward_pass
	def predict(self,x):
		print("predict")
		T=len(x)
		s,o=self.forward_pass(x)
		print("sSSSSSSSOOOOOOOOOOOOOOOOOOOOOOOOOOO")
		#print(s,o)
		return o[T-1,0]
	#计算损失 只需要计算最后那个输出与目标输出间的误差
	def calculate_loss(self,x,y):
		print("calculate_loss")
		T=len(x)
		Loss=0
		prediction=self.predict(x)
		Loss=(y-prediction)*(y-prediction)
		return Loss
	def bptt(self,x,y):
		#先计算残差项
		#输出层
		print("bptt")
		in_dim=self.in_dim
		hidden_dim=self.hidden_dim
		out_dim=self.out_dim
		T=len(x)
		s,o=self.forward_pass(x)
		kk=np.zeros((T,1))
		kk[:,0]=x
		dltakt=np.zeros((T,out_dim))
		dltaht=np.zeros((T+1,hidden_dim))
		self.dltaU=np.zeros((in_dim,hidden_dim))
		self.dltaV=np.zeros((hidden_dim,out_dim))
		self.dltaW=np.zeros((hidden_dim,hidden_dim))
		self.dltaBh=np.zeros((1,hidden_dim))
		self.dltaBk=np.zeros((1,out_dim))
		self.dltaSeta=np.zeros((1,hidden_dim))+10
		h=0
		#V
		#print(o[T-1,0],y,"sssss")
		while h<hidden_dim:
			k=0
			while k<out_dim:
				self.dltaV[h,k]=s[T-1,h]*2*(o[T-1,k]-y) 
				k=k+1
			h=h+1
		for h in np.arange(hidden_dim):
			sum=0
			for gg in np.arange(out_dim):
				sum=sum+2*(o[T-1,gg]-y)*self.V[h,gg]
			dltaht[T-1,h]=sum*(1-s[T-1,h])*(1-s[T-1,h])
		for t in np.arange(T-2)[::-1]:
			dltaht[t,h]=(1-s[t,h])*(1-s[t,h])*(np.dot(self.W[h,:].T,dltaht[t+1,:]))
		i=0
		#U
		while i<in_dim:
			h=0
			
			while h<hidden_dim:
				self.dltaU[i,h]=np.sum(kk[t,i]*dltaht[t,h]  for t in np.arange(T) )
				h=h+1
			i=i+1
		h=0
		while h<hidden_dim:
			h1=0
			while h1<hidden_dim:
				self.dltaW[h,h1]=np.sum(s[t-1,h]*dltaht[t,h1] for t in np.arange(T) )
				h1=h1+1
			h=h+1
		h=0
		while h<hidden_dim:
			self.dltaBh[0,h]=np.sum(dltaht[t,h] for t in np.arange(T) )
			h=h+1
		h=0
		while h<hidden_dim:
			self.dltaSeta[0,h]=np.sum(-dltaht[t,h] for t in np.arange(T) )
			h=h+1
		k=0
		while k<out_dim:
			self.dltaBk[0,k]=2*(o[T-1,k]-y)
			k=k+1
		return self.dltaU,self.dltaV,self.dltaW,self.dltaBh,self.dltaBk,self.dltaSeta
		
	#更新参数U,V,W n是学习率
	def update(self,X_train,Y_train,n):
		print("update")
		#model=RNNNumpy(1)#1是__init__()中in_dim参数
		size=len(X_train)
		in_dim=self.in_dim
		hidden_dim=self.hidden_dim
		out_dim=self.out_dim
		times=0
		print(self.predict(X_train[0]))
		while times<4:
			i=0
			self.dltaUsum=np.zeros((in_dim,hidden_dim))
			self.dltaVsum=np.zeros((hidden_dim,out_dim))
			self.dltaWsum=np.zeros((hidden_dim,hidden_dim))
			self.dltaBhsum=np.zeros((1,hidden_dim))
			self.dltaBksum=np.zeros((1,out_dim))
			self.dltaSetasum=np.zeros((1,out_dim))
			print("UVW---------update1")
			print(self.U,self.V,self.W)
			while i<100:
				x=X_train[i]
				y=Y_train[i]
				
				self.dltaU,self.dltaV,self.dltaW,self.dltaBh,self.dltaBk,self.dltaSeta=self.bptt(x,y)
				self.dltaUsum=self.dltaU
				self.dltaVsum=self.dltaV
				self.dltaWsum=self.dltaW
				self.dltaBhsum=self.dltaBh
				self.dltaBksum=self.dltaBk
				self.dltaSetasum=self.dltaSeta
				i=i+1
				#print("single------------------------")
				#print(self.dltaU,self.dltaV,self.dltaW,self.dltaBh,self.dltaBk)
				#print("sum---------------------")
				#print(self.dltaUsum,self.dltaVsum,self.dltaWsum,self.dltaBhsum,self.dltaBksum)
				
			#V
			h=0
			while h<hidden_dim:
				k=0
				while k<out_dim:
					self.V[h,k]=self.V[h,k]-n*self.dltaVsum[h,k]
					k=k+1
				h=h+1
			#U
			i=0
			while i<in_dim:
				h=0
				while h<hidden_dim:
					self.U[i,h]=self.U[i,h]-n*self.dltaUsum[i,h]
					h=h+1
				i=i+1
			#W
			h=0
			while h<hidden_dim:
				h1=0
				while h1<hidden_dim:
					self.W[h,h1]=self.W[h,h1]-n*self.dltaWsum[h,h1]
					h1=h1+1
				h=h+1
			h=0
			while h<hidden_dim:
				self.bh[0,h]=self.bh[0,h]-n*self.dltaBhsum[0,h]
				h=h+1
			k=0
			while k<out_dim:
				self.bk[0,k]=self.bk[0,k]-n*self.dltaBksum[0,k]
				k=k+1
			h=0
			while h<hidden_dim:
				self.seta[0,h]=self.seta[0,h]-n*self.dltaSetasum[0,h]
				h=h+1
			print("UVW---------update2")
			print(self.U,self.V,self.W,self.seta)
			print(self.predict(X_train[0]),self.predict(X_train[1]),self.predict(X_train[2]))
			print(Y_train[0],Y_train[1],Y_train[2])
			times=times+1
		
		return self.U,self.V,self.W

array_n=[4]
model=RNNNumpy(1)#1是__init__()中in_dim参数
U,V,W=model.update(X_train,Y_train,0.01)
#for i in np.arange(1):
#	print(Y_train[i],model.predict(X_train[i]))
""""for n in array_n:
	print(n)
	model=RNNNumpy(1)#1是__init__()中in_dim参数
	size=len(X_train)
	times=0
	while times<5:
		i=0
		while i<size:
			U,V,W=model.update(X_train[i],Y_train[i],n)
			
			# print("U----------------------")
			# for mm in U:
				# print(mm)
			# print("V----------------------")
			# for mm in V:
				# print(mm)
			# print("W----------------------")
			# for mm in W:
				# print(mm)
			i=i+1
		times=times+1
		print("prediction----------------------")
		for i in np.arange(10):
			print(Y_train[i],model.predict(X_train[i]))print("U----------------------")
		for mm in U:
			print(mm)
		print("V----------------------")
		for mm in V:
			print(mm)
		print("W----------------------")
		for mm in W:
			print(mm) """
	#loss=np.sum(model.calculate_loss(X_test[j],Y_test[j]) for j in np.arange(len(X_test)))
	#print(loss)
#s,o=model.forward_pass(X_train[0])
#prediction=model.predict(X_train[0])
#dltaU,dltaV,dltaW=model.bptt(X_train[0],Y_train[0])
# print("U----------------------")
# for mm in U:
	# print(mm)
# print("V----------------------")
# for mm in V:
	# print(mm)
# print("W----------------------")
# for mm in W:
	# print(mm)
# print("prediction----------------------")
#print(prediction)


