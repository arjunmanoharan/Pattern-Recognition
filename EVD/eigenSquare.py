import numpy as np
import io
import math
from PIL import Image
import os
pathname=os.path.dirname(sys.argv[0])


DIMENSION=256

def computeEigen(image):

	#--------------------perform EVD------------------#
	image=image.astype('float64')
	rec=np.zeros((DIMENSION,DIMENSION))			
	[eigen,V]=np.linalg.eig(image)	
	idx = np.argsort(eigen) 
	eigen = eigen[idx]
	V = V[:,idx]
	Vin=np.linalg.inv(V)
	sigma=np.diag(eigen)

	return np.dot(np.dot(V,sigma),Vin).real

def computeEVD(image):

	V=np.zeros((256,256,3))
	sigma=np.zeros((256,256,3))
	Vin=np.zeros((256,256,3))
	red=image[:,:,0]
	green=image[:,:,1]
	blue=image[:,:,2]
	rec=np.zeros((256,256,3))
	#-----computing the EVD channelwise-------------#
	rec[:,:,0]=computeEigen(red)
	rec[:,:,1]=computeEigen(green)
	rec[:,:,2]=computeEigen(blue)
		
	return rec


def errorVSNGraph(image,random,top):
	
	rec=np.zeros((DIMENSION,DIMENSION,3))
	rec=computeEVD(image)	
	#------------reconstruction of the image------------#
	Image.fromarray(rec.astype('uint8'),'RGB').show()
	
def main():
	
	im = Image.open(pathname+'\square.jpg').convert('RGB')
	random=True
	top=True
	a=np.asarray(im)
	errorVSNGraph(a,random,top)
		
main()

