import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thisdir+'/HW1_files')
sys.path.append(thisdir+'/HW1_files/prokudin-gorskii/')

import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


def checkoffset(imageA,imageB):
	# measure_old =100000000
	measure_old=0
	imageA = imageA.astype('float')
	imageB = imageB.astype('float')

	for i in range(-15,15):
		for j in range(-15,15):

			moved_image_right = np.roll(imageA, i, axis=1) #right
			# measure=np.dot(moved_image,imageB.T)
			moved_image_up = np.roll(moved_image_right, j, axis=0) #up
			# measure=np.sum(np.corrcoef(moved_image_up,imageB))
			measure= np.sum(moved_image_up*imageB)

			# measure=np.sum(np.corrcoef(moved_image_up,imageB))
			if measure>measure_old:
				x_off = i
				y_off = j
				measure_old = measure
				# if i %5 ==0:
					# print(i,measure)
	# print('sameimage',np.sum(imageB*imageB))

	return x_off, y_off

# def checkoffset_recursive(imageA,imageB):

# 	shrinked_imageA = cv2.resize(imageA,(w//2,h//2))
# 	shrinked_imageB = cv2.resize(imageB,(w//2,h//2))

# 	shrinked_xoff, shrinked_yoff = checkoffset(shrinked_imageA,shrinked_imageB)

# 	imageA_right =  np.roll(imageA,shrinked_xoff*2, axis=1)# aligned
# 	imageA_up =  np.roll(imageA_right, shrinked_yoff*2, axis=0)# aligned

# 	x_off, y_off = checkoffset(imageA_up,imageB)

# 	return x_off, y_off


def checkoffset_recursive(imageA,imageB,xoff,yoff,n):
	if n == 1:
		x_off, y_off = checkoffset(imageA,imageB)
		x_off_final = xoff + x_off
		y_off_final = yoff + y_off
		return x_off_final, y_off_final
	else:
 		shrinked_imageA = cv2.resize(imageA,(h//2,w//2))
 		shrinked_imageB = cv2.resize(imageB,(h//2,w//2))

 		shrinked_xoff, shrinked_yoff = checkoffset(shrinked_imageA,shrinked_imageB)
 		print('middle',shrinked_xoff, shrinked_yoff )

 		imageA_right =  np.roll(imageA,shrinked_xoff, axis=1)# aligned
 		imageA_up =  np.roll(imageA_right, shrinked_yoff, axis=0)# aligned
 		# plt.imshow(imageA_up)
 		# plt.show()

 		n = n-1
 		shrinked_xoff += xoff 
 		shrinked_yoff += yoff 

 		return checkoffset_recursive(imageA_up,imageB,shrinked_xoff,shrinked_yoff,n)


def stackimages(image1,image2,image3):
	image1_b = image1[:,:,None]
	image2_g = image2[:,:,None]
	image3_r = image3[:,:,None]
	stacked=np.dstack((image1_b,image2_g))
	stacked_all = np.dstack((stacked,image3_r))
	stacked_all_rgb = cv2.cvtColor(stacked_all, cv2.COLOR_BGR2RGB)

	return stacked_all_rgb


image = plt.imread('../HW1_files/seoul_tableau.jpg')
# image = plt.imread('../prokudin-gorskii/00125v.jpg')

h,w = image.shape

# plt.imshow(shrinked_image)



h_ind =round(h/3)
image1 = image[:h_ind,:-10]
# image1=  np.roll(image1,(12,8), axis=(1,0))# aligned
# image1=  np.roll(image1,(1,-2), axis=(1,0))# aligned

# image1_b = image1[:,:,None]

image2 = image[h_ind:2*h_ind,:-10]
# image2 =  np.roll(image2,(22,-1), axis=(1,0))# aligned
# image2 =  np.roll(image2,(6,0), axis=(1,0))# aligned
# image2_g = image2[:,:,None]

image3 = image[2*h_ind:3*h_ind,:-10]
# image3_r = image3[:,:,None]

# stacked=np.dstack((image1_b,image2_g))
# stacked_all = np.dstack((stacked,image3_r))
# stacked_all_rgb = cv2.cvtColor(stacked_all, cv2.COLOR_BGR2RGB)

stacked = stackimages(image1,image2,image3)

plt.imshow(stacked)
plt.show()
# plt.imsave('test_prob3.png',stacked_all_rgb)


# x_off, y_off=checkoffset(image1,image3)
x_off, y_off=checkoffset_recursive(image2,image3,0,0,2)
print(x_off, y_off)


image1=  np.roll(image1,(12,8), axis=(1,0))# aligned
image2 =  np.roll(image2,(6,0), axis=(1,0))





##
# shrinked_image1 = cv2.resize(image1,(h//2,w//2))
# shrinked_image1=  np.roll(shrinked_image1,(12,6), axis=(1,0))# aligned
# shrinked_image1_b = shrinked_image1[:,:,None]


# shrinked_image2 = cv2.resize(image2,(h//2,w//2))
# # shrinked_image2=  np.roll(shrinked_image2,(14+8,-1), axis=(1,0))# aligned
# shrinked_image2=  np.roll(shrinked_image2,(14+8,-1), axis=(1,0))# aligned
# shrinked_image2_g = shrinked_image2[:,:,None]


# shrinked_image3 = cv2.resize(image3,(h//2,w//2))
# shrinked_image3_r = shrinked_image3[:,:,None]

# shrinked_stacked=np.dstack((shrinked_image1_b,shrinked_image2_g))
# shrinked_stacked_all = np.dstack((shrinked_stacked,shrinked_image3_r))
# shrinked_stacked_all_rgb = cv2.cvtColor(shrinked_stacked_all, cv2.COLOR_BGR2RGB)

# plt.imshow(shrinked_stacked_all_rgb)
# plt.show()

# shrinked_xoff, shrinked_yoff = checkoffset(shrinked_image2,shrinked_image3)
# print(shrinked_xoff, shrinked_yoff)




