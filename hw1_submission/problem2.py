
import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(thisdir)
sys.path.append(thisdir+'/prokudin-gorskii/')

import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

# image = plt.imread('../prokudin-gorskii/efros_tableau.jpg')
# image = plt.imread('../efros_tableau.jpg')
image = plt.imread('../HW1_files/prokudin-gorskii/00398v.jpg')

h,w = image.shape

h_ind =round(h/3)
image1 = image[:h_ind,:]
image2 = image[h_ind:2*h_ind,:]
image3 = image[2*h_ind:3*h_ind,:]
# image1 = image[20:338,22:380]
#efros_tableau.jpg
# image1 = image[:421,:]

#00125v.jpg
# image1 =  np.roll(image1,-1, axis=1)# aligned
# image1 =  np.roll(image1,9, axis=0)# aligned

#00153v.jpg
# image1 =  np.roll(image1,-6, axis=1)# aligned
# image1 =  np.roll(image1,5, axis=0)# aligned

# 00351v.jpg
# image1 =  np.roll(image1,-1, axis=1)# aligned 
# image1 =  np.roll(image1,6, axis=0)# aligned

#00398v.jpg
# image1 =  np.roll(image1,0, axis=1)# aligned
# image1 =  np.roll(image1,8, axis=0)# aligned

#01112v.jpg
# image1 =  np.roll(image1,-1, axis=1)# aligned
# image1 =  np.roll(image1,14, axis=0)# aligned

#efros_tableau.jpg
# image1 =  np.roll(image1,5, axis=1)# aligned
# image1 =  np.roll(image1,0, axis=0)# aligned

image1_b = image1[:,:,None]

# image2 = image[342+6:673-7,22:380]
#efros_tableau.jpg
# image2 = image[421:842,:]

#00153v.jpg
# image2 =  np.roll(image2,-2, axis=1) #aligned
# image2 =  np.roll(image2,-1, axis=0) #aligned


#00125v.jpg
# image2 =  np.roll(image2,0, axis=1) #aligned
# image2 =  np.roll(image2,2, axis=0) #aligned

# 00351v.jpg
# image2 =  np.roll(image2,-1, axis=1) #aligned
# image2 =  np.roll(image2,-3, axis=0) #aligned up down

#00398v.jpg
# image2 =  np.roll(image2,-1, axis=1) #aligned
# image2 =  np.roll(image2,0, axis=0) #aligned up down

#01112v.jpg
# image2 =  np.roll(image2,-1, axis=1) #aligned
# image2 =  np.roll(image2,1, axis=0) #aligned up down

#efros_tableau.jpg
# image2 =  np.roll(image2,10, axis=1) #aligned
# image2 =  np.roll(image2,0, axis=0) #aligned up down

image2_g = image2[:,:,None]

# image3 = image[682+1:1002-1,22:380]
#efros_tableau.jpg
# image3 = image[842:1264,:]

image3_r = image3[:,:,None]

stacked=np.dstack((image1_b,image2_g))
stacked_all = np.dstack((stacked,image3_r))
stacked_all_rgb = cv2.cvtColor(stacked_all, cv2.COLOR_BGR2RGB)

plt.imshow(stacked_all_rgb )
plt.show()


# task 2

def checkoffset(imageA,imageB):
	# measure_old =100000000
	measure_old=0
	imageA = imageA.astype('float')
	imageB = imageB.astype('float')

	for i in range(-15,15):
		# if i<15:
		# 	moved_image = np.pad(imageA[:,i:], ((0,0), (0,i)),'constant')
				
		# elif i == 15:
		# 	moved_image=imageA

		# else:	
		# 	moved_image = np.pad(imageA[:,:-i+15], ((0,0), (i-15,0)),'constant')

		for j in range(-15,15):
		# 	if j<15:
		# 		moved_image_up = np.pad(moved_image[j:,:], ((0,j), (0,0)),'constant')

		# 	elif j == 15:
		# 		moved_image_up=moved_image

		# 	else:	
		# 		moved_image_up = np.pad(moved_image[:-j+15,:], ((j-15,0), (0,0)),'constant')


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
	# return x_off


x_offset,y_offset=checkoffset(image2,image3)

print('output',x_offset,y_offset)