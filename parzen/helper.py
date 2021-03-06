from __future__ import division
import numpy as np
import math

pathname= os.path.dirname(sys.argv[0])

def readData():

	#--------------reading from file--------------------#
	valid0=np.load(pathname+'/data/image/feats/processed/validation0.npy')
	valid1=np.load(pathname+'/data/image/feats/processed/validation1.npy')
	valid2=np.load(pathname+'/data/image/feats/processed/validation2.npy')

	test0=np.load(pathname+'/data/image/feats/processed/test0.npy')
	test1=np.load(pathname+'/data/image/feats/processed/test1.npy')
	test2=np.load(pathname+'/data/image/feats/processed/test2.npy')

	train0=np.load(pathname+'/data/image/feats/processed/train0.npy')
	train1=np.load(pathname+'/data/image/feats/processed/train1.npy')
	train2=np.load(pathname+'/data/image/feats/processed/train2.npy')
	
	mean=np.zeros((3,23))	
	

	t0=train0.shape[0]
	t1=train1.shape[0]
	t2=train2.shape[0]

	trai0=np.vstack((train0,valid0))
	trai1=np.vstack((train1,valid1))
	trai2=np.vstack((train2,valid2))
	

	w=train0.shape[0]+train1.shape[0]+train2.shape[0]

	Q=np.vstack((train0,np.vstack((train1,train2))))		

	b=t0+valid0.shape[0]+test0.shape[0]
	c=b+t1+valid1.shape[0]
	d=c+test1.shape[0]

	train0=Q[0:t0+valid0.shape[0],:]
	train1=Q[b:b+t1+valid1.shape[0],:]
	train2=Q[d:d+t2+valid2.shape[0]]
	

	test0=Q[train0.shape[0]:train0.shape[0]+test0.shape[0]]
	test1=Q[train0.shape[0]+test0.shape[0]+train1.shape[0]:train0.shape[0]+test0.shape[0]+train1.shape[0]+test1.shape[0]]
	test2=Q[Q.shape[0]-test2.shape[0]:]
	
	#-----------------Performing variance normalization-------------#
	train0=(train0)/np.var(Q)	
	train1=(train1)/np.var(Q)	
	train2=(train2)/np.var(Q)

	

	a=test0.shape[0]
	b=test1.shape[0]
	c=test2.shape[0]
	return train0,test0,train1,test1,train2,test2,a,b,c