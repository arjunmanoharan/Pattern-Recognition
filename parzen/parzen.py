from __future__ import division
import numpy as np
import math
import helper
pathname= os.path.dirname(sys.argv[0])



#-------------calculating the prob test point belonging to a class----------------#

def buildGaussian(test,train0,train1,train2):
	
	h=0.01*np.identity(23)

	prob=np.zeros(3)
	det=np.linalg.det(h)

	const=math.pow(2*math.pi,train0.shape[1]/2)
	covinv=np.linalg.inv(h)
	
	for i in range(train0.shape[0]):
		diff=test-train0[i]
		sum1=np.dot(np.transpose(diff),covinv)
		prob[0]=prob[0]+(1/const)*np.exp((-0.5)*(np.sum(np.dot(sum1,diff),0)))
	prob[0]=prob[0]/train0.shape[0]

	for i in range(train1.shape[0]):
		diff=test-train1[i]
		sum1=np.dot(np.transpose(diff),covinv)
		prob[1]=prob[1]+(1/const)*np.exp((-0.5)*(np.sum(np.dot(sum1,diff),0)))
	prob[1]=prob[1]/train1.shape[0]

	for i in range(train2.shape[0]):
		diff=test-train2[i]
		sum1=np.dot(np.transpose(diff),covinv)
		prob[2]=prob[2]+(1/const)*np.exp((-0.5)*(np.sum(np.dot(sum1,diff),0)))
	prob[2]=prob[2]/train2.shape[0]
	
	return prob

if name=='__main__':
	

	train0,test0,train1,test1,train2,test2,a,b,c=helper.readData()
	train=np.vstack((train0,np.vstack((train1,train2))))
	test=np.vstack((test0,np.vstack((test1,test2))))
	
	t=int(math.ceil(a/36))+int(math.ceil(b/36))+int(math.ceil(c/36))
	label=np.zeros(t)
	label[:int(math.ceil(a/36))]=0
	label[int(math.ceil(a/36)):int(math.ceil(a/36))+int(math.ceil(b/36))]=1
	label[int(math.ceil(a/36))+int(math.ceil(b/36)):]=2

	print "l,",label.shape
	point=np.zeros(131)
	pro=np.zeros((131,3))
	
	t0=train0.shape[0]
	t1=train0.shape[0]+train1.shape[0]
	accuracy=k=count=0
	
	#------------deciding the class a point belongs to-----------------------#
	
	for i in range(131):
		
		for j in range(36):
			
			temp=buildGaussian(test[i*36+j],train0,train1,train2)
			if(temp[0]<1e-300):
				temp[0]=-300
			if(temp[1]<1e-300):
				temp[1]=-300
			if(temp[2]<1e-300):
				temp[2]=-300
			pro[i]=pro[i]+np.log(temp)	
		point[i]=np.argmax(pro[i,:])	
		
	
	
	f= open(pathname+'/pr1.txt','w')
	for i in range(label.shape[0]):
		
		f.write(str(int(label[i])))
		f.write(' ')
		f.write(str(pro[i,0]))
		f.write(' ')
		f.write(str(pro[i,1]))
		f.write(' ')
		f.write(str(pro[i,2]))
		f.write('\n')
	print count
	print point

	#-----------calculate accuracy-------------#
	for i in range(131):
		if point[i]==label[i]:
			accuracy=accuracy+1
	print accuracy,(accuracy*100)/131
	
