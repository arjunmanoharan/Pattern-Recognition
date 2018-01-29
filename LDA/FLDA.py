from __future__ import division
import numpy as np
import scipy.linalg as sp
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from scipy.stats import multivariate_normal
import helper
pca_comp=12

def computeLDA(train0,train1,train2,noOfClasses,shape):
	
	mean=np.zeros((noOfClasses,shape))
	S=np.zeros((noOfClasses,shape,shape))
	Sw=np.zeros((shape,shape))
	Sb=np.zeros((shape,shape))
	data=np.vstack((train0,np.vstack((train1,train2))))	

	mean[0]=np.mean(train0,0)
	mean[1]=np.mean(train1,0)
	mean[2]=np.mean(train2,0)

	

	#------------performing variance normalization-----------------#
	diff=train0-mean[0]
	print "s",diff.shape
	S[0]=np.dot(np.transpose(diff),diff)
	S[0]=S[0]/np.var(train0)
	S[0]=whitening(S[0])
		

	diff=np.asmatrix(train1-mean[1])
	S[1]=np.dot(np.transpose(diff),diff)
	S[1]=S[1]/np.var(train1)
	S[1]=whitening(S[1])
	diff=np.asmatrix(train2-mean[2])
	S[2]=np.dot(np.transpose(diff),diff)
	S[2]=S[2]/np.var(train2)
	S[2]=whitening(S[2])
	
	Sw=S[0]+S[1]+S[2]

	
	meanTotal=np.mean(data,0)
	
	diff=np.transpose(np.matrix(mean[0]-meanTotal))	
	S[0]=train0.shape[0]*np.dot(diff,np.transpose(diff))
	
	diff=np.transpose(np.matrix(mean[1]-meanTotal))	
	S[1]=train1.shape[0]*np.dot(diff,np.transpose(diff))

	diff=np.transpose(np.matrix(mean[2]-meanTotal))	
	S[2]=train2.shape[0]*np.dot(diff,np.transpose(diff))
	
	
	Sb=S[0]+S[1]+S[2]
	
	
	vals,weight=sp.eigh(np.dot(sp.inv(Sw),Sb),eigvals=(data.shape[1]-2,data.shape[1]-1))
	return weight

def whitening(X):
	fudge=1E-18
	Xcov = np.dot(X.T,X)

	# eigenvalue decomposition of the covariance matrix
	d, V = np.linalg.eigh(Xcov)

	# a fudge factor can be used so that eigenvectors associated with
	# small eigenvalues do not get overamplified.
	D = np.diag(1. / np.sqrt(d+fudge))

	# whitening matrix
	W = np.dot(np.dot(V, D), V.T)

	# multiply by the whitening matrix
	X_white = np.dot(X, W)
	return X_white

def gaussian(data,mean,cov):
	#--------------------computing gaussian for the given mean and covariance----------------#
	prob=np.zeros(data.shape[0])
	det=np.linalg.det(cov)
	
	const=math.pow(2*math.pi,data.shape[1]/2)
	
	covinv=sp.inv(cov)
	for i in range(data.shape[0]):

		diff=data[i]-mean
		mul=np.dot(covinv,np.transpose(diff))
		sum1=np.dot(diff,mul)
		prob[i]=np.exp((-0.5)*(np.sum(sum1,0)))
	
		prob[i]=prob[i]/(const*math.sqrt(det))
	return prob	

def test(train0,train1,train2,test0,test1,test2,label,a,b,c):

	data=np.vstack((test0,np.vstack((test1,test2))))	

	a=test0.shape[0]
	b=test1.shape[0]
	c=test2.shape[0]

	mean1=np.mean(train0,0)
	mean2=np.mean(train1,0)
	mean3=np.mean(train2,0)

	
	cov1=np.cov(np.transpose(train0))
	cov2=np.cov(np.transpose(train1))
	cov3=np.cov(np.transpose(train2))

	prob=np.zeros((data.shape[0],3))
	prob[:,0]=gaussian(data,mean1,cov1)
	prob[:,1]=gaussian(data,mean2,cov2)
	prob[:,2]=gaussian(data,mean3,cov3)
	print prob[:,0]
	
	count=0
	t=int(math.ceil(a/36))+int(math.ceil(b/36))+int(math.ceil(c/36))
	label=np.zeros(t)
	label[:int(math.ceil(a/36))]=0
	label[int(math.ceil(a/36)):int(math.ceil(a/36))+int(math.ceil(b/36))]=1
	label[int(math.ceil(a/36))+int(math.ceil(b/36)):]=2
	pro=np.zeros((t,3))
	index=np.zeros(label.shape[0])
	k=0
	for i in range(data.shape[0]):
	
		if i%36 != 0 or i==0:
			if(prob[i,0]<1e-300):
				prob[i,0]=1
			if(prob[i,1]<1e-300):
				prob[i,1]=1
			if(prob[i,2]<1e-300):
				prob[i,2]=1
			pro[k,0]=pro[k,0]+math.log(prob[i,0])
			pro[k,1]=pro[k,1]+math.log(prob[i,1])
			pro[k,2]=pro[k,2]+math.log(prob[i,2])
		else:
		
			index[count]=np.argmax(pro[k])
			k=k+1
			count=count+1
	print pro
			
	inde=np.zeros((3,pro.shape[0],pro.shape[1]))

	for j in range(3):
		inde[0,:,j]=pro[:,j]-pro[:,0]

	for j in range(3):
		inde[1,:,j]=pro[:,j]-pro[:,1]

	for j in range(3):
		inde[2,:,j]=pro[:,j]-pro[:,2]

	

	p=np.exp(inde)
	
	for k in range(3):
		for j in range(pro.shape[0]):
			pro[j,k]=1/np.sum(p[k,j,:],0)
		
	
	label=np.transpose(np.asmatrix(label))
	
	tab=np.hstack((label,pro))
	f= open(pathname+'/lda'+`pca_comp`+'.txt','w')
	for i in range(tab.shape[0]):
		
		f.write(str(int(tab[i,0])))
		f.write(' ')
		f.write(str(tab[i,1]))
		f.write(' ')
		f.write(str(tab[i,2]))
		f.write(' ')
		f.write(str(tab[i,3]))
		f.write('\n')
	
	acc=0
	
	for i in range(label.shape[0]):
		if index[i]==label[i]:
			acc=acc+1
	
	print acc,(acc*100)/label.shape[0],label.shape[0]
	
if name=='__main__':
	
	train0,test0,train1,test1,train2,test2,a,b,c=helper.readDate()
	
	weight=computeLDA(train0,train1,train2,3,train0.shape[1])
	label=1
	print weight.shape,train0.shape

	#-----------------plotting graphs-------------------------#
	data0=np.dot(train0,weight)
	data1=np.dot(train1,weight)
	data2=np.dot(train2,weight)
	plt.plot(data0[:,0],data0[:,1],'go')
	plt.plot(data1[:,0],data1[:,1],'bo')
	plt.plot(data2[:,0],data2[:,1],'ro')
	#plt.show()
	test0=np.dot(test0,weight)
	test1=np.dot(test1,weight)
	test2=np.dot(test2,weight)
	plt.savefig(pathname+'/plots/LDA/dim.jpeg')
	accuracy=test(data0,data1,data2,test0,test1,test2,label,a,b,c)
	
