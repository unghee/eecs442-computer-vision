
import os,sys


import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


def stackimages(image1,image2,image3):
	image1_b = image1[:,:,None]
	image2_g = image2[:,:,None]
	image3_r = image3[:,:,None]
	stacked=np.dstack((image1_b,image2_g))
	stacked_all = np.dstack((stacked,image3_r))
	stacked_all_rgb = cv2.cvtColor(stacked_all, cv2.COLOR_BGR2RGB)

	return stacked_all_rgb
def checkoffset_recursive(imageA,imageB,xoff,yoff,n,h,w):
	if n == 1:
		x_off, y_off = checkoffset(imageA,imageB)
		x_off_final = xoff + x_off
		y_off_final = yoff + y_off
		return x_off_final, y_off_final
	else:
 		shrinked_imageA = cv2.resize(imageA,(h//2,w//2))
 		shrinked_imageB = cv2.resize(imageB,(h//2,w//2))

 		shrinked_xoff, shrinked_yoff = checkoffset(shrinked_imageA,shrinked_imageB)
 		# print('middle',shrinked_xoff, shrinked_yoff )

 		imageA_right =  np.roll(imageA,shrinked_xoff, axis=1)# aligned
 		imageA_up =  np.roll(imageA_right, shrinked_yoff, axis=0)# aligned
 		n = n-1
 		shrinked_xoff += xoff 
 		shrinked_yoff += yoff 

 		return checkoffset_recursive(imageA_up,imageB,shrinked_xoff,shrinked_yoff,n,h,w)

def checkoffset(imageA,imageB):
	# measure_old =100000000
	measure_old=0
	imageA = imageA.astype('float')
	imageB = imageB.astype('float')

	for i in range(-15,15):
		for j in range(-15,15):

			moved_image_right = np.roll(imageA, i, axis=1) #right
			moved_image_up = np.roll(moved_image_right, j, axis=0) #up
			measure= np.sum(moved_image_up*imageB)

			if measure>measure_old:
				x_off = i
				y_off = j
				measure_old = measure

	return x_off, y_off
