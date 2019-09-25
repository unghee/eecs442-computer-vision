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


image = plt.imread('../HW1_files/seoul_tableau.jpg')

h,w = image.shape

h_ind =round(h/3)
image1 = image[:h_ind,:-10]
image2 = image[h_ind:2*h_ind,:-10]
image3 = image[2*h_ind:3*h_ind,:-10]


stacked = stackimages(image1,image2,image3)

plt.imshow(stacked)
plt.show()
# plt.imsave('test_prob3.png',stacked_all_rgb)


x_off1, y_off1=checkoffset_recursive(image1,image3,0,0,2,h,w)
x_off2, y_off2=checkoffset_recursive(image2,image3,0,0,2,h,w)
print(x_off1, y_off1)
print(x_off2, y_off2)

image1=  np.roll(image1,(x_off1, y_off1), axis=(1,0))# aligned
image2 =  np.roll(image2,(x_off2, y_off2), axis=(1,0))

stacked = stackimages(image1,image2,image3)

plt.imshow(stacked)
plt.show()

plt.imsave('Problem3_task3_2.png',stacked)



