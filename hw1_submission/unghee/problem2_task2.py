import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thisdir+'/HW1_files')
sys.path.append(thisdir+'/HW1_files/prokudin-gorskii/')

import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from utils import *


# image = plt.imread('../HW1_files/prokudin-gorskii/00149v.jpg')
# image = plt.imread('../HW1_files/seoul_tableau.jpg')
imagename = ['00125v','00149v','00153v','00351v','00398v','01112v','efros_tableau']

for i in range(len(imagename)):
	image = plt.imread('../HW1_files/prokudin-gorskii/'+ imagename[i]+'.jpg')

	if imagename[i] == 'efros_tableau':
		#Automatically crop borders
		image = image[(image.mean(axis=1)<240 ),:]
		image = image[:,(image.mean(axis=0)<240)&(image.mean(axis=0)>30)]
	else:	
		image = image[(image.mean(axis=1)<230 ),:]
		image = image[:,(image.mean(axis=0)<230)&(image.mean(axis=0)>90)]
	h,w = image.shape

	h_ind =round(h/3)
	image1 = image[:h_ind,:]
	image2 = image[h_ind:2*h_ind,:]
	image3 = image[2*h_ind:3*h_ind,:]
	if 3*h_ind>h:
		image3 = image[2*h_ind-1:3*h_ind-1,:]

	# image1 = image[20:338,22:380]
	# image2 = image[342+6:673-7,22:380]
	# image3 = image[682+1:1002-1,22:380]

	stacked = stackimages(image1,image2,image3)

	plt.imshow(stacked)
	plt.show()
	# plt.imsave('test_prob3.png',stacked_all_rgb)

	# image1=  np.roll(image1,(2, -8), axis=(1,0))
	# image2 =  np.roll(image2,(1, 0), axis=(1,0))

	x_off1, y_off1=checkoffset(image1,image3)
	x_off2, y_off2=checkoffset(image2,image3)
	# x_off1, y_off1=checkoffset_recursive(image1,image3,0,0,2,h,w)
	# x_off2, y_off2=checkoffset_recursive(image2,image3,0,0,2,h,w)
	print(x_off1, y_off1)
	print(x_off2, y_off2)

	image1=  np.roll(image1,(x_off1, y_off1), axis=(1,0))# aligned
	image2 =  np.roll(image2,(x_off2, y_off2), axis=(1,0))

	stacked = stackimages(image1,image2,image3)

	plt.imshow(stacked)
	plt.show()

	# plt.imsave(imagename[i]+'stacked_alinged'+'.jpg',stacked)



