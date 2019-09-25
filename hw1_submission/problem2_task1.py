
import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(thisdir+'/HW1_files')
sys.path.append(thisdir+'/HW1_files/prokudin-gorskii/')

import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from utils import *

# image = plt.imread('../HW1_files/prokudin-gorskii/efros_tableau.jpg')
imagename = ['00125v','00149v','00153v','00351v','00398v','01112v']

for i in range(len(imagename)):
	image = plt.imread('../HW1_files/prokudin-gorskii/'+ imagename[i]+'.jpg')


	h,w = image.shape

	h_ind =round(h/3)
	image1 = image[:h_ind,:]
	image2 = image[h_ind:2*h_ind,:]
	image3 = image[2*h_ind:3*h_ind,:]


	stacked = stackimages(image1,image2,image3)

	# plt.imshow(stacked)
	# plt.show()
	plt.imsave(imagename[i]+'stacked'+'.jpg',stacked)