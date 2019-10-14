import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(thisdir+'/hw2_files')


import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from filters import *
# from utils import *


image = plt.imread('../hw2_files/'+ 'grace_hopper'+'.png')
shape_image=np.shape(image)

patched_images=image_patches(image)

fig, axs = plt.subplots(2,37 , figsize=(9, 3), sharey=True)

for i in range(37):
	# axs[i].imshow(patched_images[i], cmap='gray')
	for j in range(2):
		# axs[j][i].imshow(patched_images[i+j*24], cmap='gray',vmin=-5, vmax=5)
		axs[j][i].imshow(patched_images[i+j*37], cmap='gray')
	# axs[1].imshow(patched_images[1], cmap='gray')
	# axs[2].imshow(patched_images[2], cmap='gray')
# plt.imshow(patched_images[0], cmap='gray')
# sliced_list =[]
# sliced_list.append((image[0:16*(1),16*0:16*(1)]-np.mean(image))/np.std(image))

# plt.imshow(sliced_list[0], cmap='gray',vmin=-0.5, vmax=0.5)
# plt.imshow(patched_images[0], cmap='gray',vmin=-2.5, vmax=2.5)

# plt.savefig(patched_images[0])

# plt.show()
# plt.imsave('problem1.png',patched_images[0], cmap = 'gray')
fig.savefig('problem1.png')


# fig2, axs = plt.subplots(1,3 , figsize=(9, 3), sharey=True)


# axs[0].imshow(patched_images[5], cmap='gray')
# axs[1].imshow(patched_images[8], cmap='gray')
# axs[2].imshow(patched_images[15], cmap='gray')
# fig2.savefig('problem1.png')

# xint = ((x-xmin(x))/(max(x)-min(x))*255).astype(np.uint8)

