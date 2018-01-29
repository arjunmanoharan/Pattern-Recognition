import numpy as np
import scipy.linalg as sp
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
def readDate():
	file_list=os.listdir('/home/cs17s004/data/speech/isolated/20/1')
	c=0
	
	data1=np.zeros((c,38),dtype='float64')			
	
	data1len=np.zeros(len(file_list))
	for i in range(len(file_list)):
		f=open('/home/cs17s004/data/speech/isolated/20/1/'+file_list[i],'r')
		a=f.readline().strip().split(' ')

		dummy=np.zeros((int(a[1]),38),dtype='float64')
		
		for j in range(int(a[1])):
			dummy[j]=f.readline().strip().split(' ')
		data1len[i]=int(a[1])
		if(c==0):
			data1=dummy
		else:
			data1=np.vstack((data1,dummy))	
		c=c+int(a[1])
	

###############################################################################


	data5=np.zeros((c,38),dtype='float64')			
	c=0
	file_list=os.listdir('/home/cs17s004/data/speech/isolated/20/5')
	data5len=np.zeros(len(file_list))
	for i in range(len(file_list)):
		f=open('/home/cs17s004/data/speech/isolated/20/5/'+file_list[i],'r')
		a=f.readline().strip().split(' ')

		dummy=np.zeros((int(a[1]),38),dtype='float64')
		
		for j in range(int(a[1])):
			dummy[j]=f.readline().strip().split(' ')
		data5len[i]=int(a[1])
		
		if(c==0):
			data5=dummy
		else:
			data5=np.vstack((data5,dummy))	
		c=c+int(a[1])
	print(data5.shape)
	

#################################################################################


	dataZ=np.zeros((c,38),dtype='float64')			
	c=0
	file_list=os.listdir('/home/cs17s004/data/speech/isolated/20/z')
	dataZlen=np.zeros(len(file_list))

	for i in range(len(file_list)):
		f=open('/home/cs17s004/data/speech/isolated/20/z/'+file_list[i],'r')
		a=f.readline().strip().split(' ')

		dummy=np.zeros((int(a[1]),38),dtype='float64')
		
		for j in range(int(a[1])):
			dummy[j]=f.readline().strip().split(' ')
		dataZlen[i]=int(a[1])
		if(c==0):
			dataZ=dummy
		else:
			dataZ=np.vstack((dataZ,dummy))	
		c=c+int(a[1])
	
	#variance normalization
	data=np.vstack((data1,np.vstack((data5,dataZ))))
	pca=PCA(n_components=38)
	#print data
	
	#print dat

	#print np.var(data1,0)
	plt.plot(data1[:,36],data1[:,37],'rx')
	plt.plot(data5[:,36],data5[:,37],'bx')
	plt.plot(dataZ[:,36],dataZ[:,37],'gx')
	#plt.show()
	#Q,R=np.linalg.qr(data)
	Q=pca.fit_transform(data)
	print Q.shape
	#print R.shape
	'''data1=Q[0:data1.shape[0],:]
	data5=Q[data1.shape[0]:data1.shape[0]+data5.shape[0],:]
	dataZ=Q[data5.shape[0]+data1.shape[0]:,:]
	'''
	
	#print np.var(data1,0)
	
	data1=(data1-np.mean(data1,0))/np.var(data1,0)	
	data5=(data5-np.mean(data5,0))/np.var(data5,0)	
	dataZ=(dataZ-np.mean(dataZ,0))/np.var(dataZ,0)	
	#print sp.det(np.cov(data1))
	'''
	data1=(data1)/np.var(data1,0)	
	data5=(data5)/np.var(data5,0)	
	dataZ=(dataZ)/np.var(dataZ,0)
	'''
	
	'''
	plt.plot(data1[:,36],data1[:,37],'ro')
	plt.plot(data5[:,36],data5[:,37],'bo')
	plt.plot(dataZ[:,36],dataZ[:,37],'go')
	plt.show()
	'''
	return data1,data5,dataZ,data1len,data5len,dataZlen	

def computeLDA(dataDict,noOfClasses,shape):
	
	mean=np.zeros((noOfClasses,shape))
	S=np.zeros((noOfClasses,shape,shape))
	Sw=np.zeros((shape,shape))
	Sb=np.zeros((shape,shape))

	for i in range(noOfClasses):
		mean[i]=np.mean(dataDict[`i`],0)
		
		if i == 0:
			data=dataDict[`i`]
		else:
			data=np.vstack((data,dataDict[`i`]))	
		diff=np.asmatrix(dataDict[`i`]-mean[i])
		S[i]=np.dot(np.transpose(diff),diff)
		Sw=Sw+S[i]

	
	meanTotal=np.mean(data,0)
	

	for i in range(noOfClasses):
		diff=np.asmatrix(mean[i]-meanTotal)
		data=dataDict[`i`]
		S[i]=data.shape[0]*np.dot(np.transpose(diff),diff)
		
		Sb=Sb+S[i]

	#print Sb
	vals,weight=sp.eigh(np.dot(sp.inv(Sw),Sb),eigvals=(35,37))

	#print weight.shape,vals
	return weight
def gaussian(data):
	mean=np.mean(data,0)
	print "sh=",mean.shape()
	cov=np.cov(np.transpose(data))
	#cov=np.diag(cov)
	#cov=np.diagflat(cov[:,0])
	

	det=np.linalg.det(cov)
	const=2*math.pi
	diff=data-mean
	print len(diff)
	print det
	
	sum1=np.asmatrix(np.dot(diff,np.dot(np.linalg.inv(cov),np.transpose(diff))))
	print "sum1=",sum1
	prob=np.exp((-0.5)*(np.sum(sum1,0)))
	print prob
	prob=prob/(const*math.sqrt(det))

	
	return prob

def test(projectedData,label):


	data1=projectedData['0']
	data2=projectedData['1']
	data3=projectedData['2']
	gaussian(data1)
	plt.plot(data1[:,0],data1[:,1],'ro')
	plt.plot(data2[:,0],data2[:,1],'bo')
	plt.plot(data3[:,0],data3[:,1],'go')
	#plt.show()
def main():
	
	data1,data5,dataZ,data1len,data5len,dataZlen=readDate()
	
	dictionaryData=dict()

	dictionaryData['0']=data1
	dictionaryData['1']=data5
	dictionaryData['2']=dataZ
	
	weight=computeLDA(dictionaryData,3,data1.shape[1])

	projectedData=dict()

	projectedData['0']=np.dot(dictionaryData['0'],weight)
	projectedData['1']=np.dot(dictionaryData['1'],weight)
	projectedData['2']=np.dot(dictionaryData['2'],weight)
	label=1
	accuracy=test(projectedData,label)

	#print accuracy
	

main()