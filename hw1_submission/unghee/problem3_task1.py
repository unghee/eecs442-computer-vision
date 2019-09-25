import cv2
import matplotlib.pyplot as plt

import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(thisdir+'/HW1_files')
sys.path.append(thisdir+'/HW1_files/prokudin-gorskii/')

class imageplot:
	def __init__(self,filename,colorA,colorB,colorC):
		self.filename = filename
		self.colorA = colorA
		self.colorB = colorB
		self.colorC = colorC

	def plotting(self,LAB):
		image = cv2.imread(self.filename)		

		if LAB:
			lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
			b,g,r = cv2.split(lab)
		else:
			b,g,r = cv2.split(image)
		


		fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

		axs[0].imshow(b, cmap='gray')
		axs[1].imshow(g, cmap='gray')
		axs[2].imshow(r, cmap='gray')

		axs[0].set_title(self.colorA)
		axs[1].set_title(self.colorB)
		axs[2].set_title(self.colorC)

		fig.suptitle(self.colorA +self.colorB +self.colorC +'Color Space '+self.filename)


i1=imageplot('../HW1_files/indoor.png','B','G','R')
i1.plotting(False)

i2=imageplot('../HW1_files/outdoor.png','B','G','R')
i2.plotting(False)

i2=imageplot('../HW1_files/indoor.png','L','A','B')
i2.plotting(True)
i2=imageplot('../HW1_files/outdoor.png','L','A','B')
i2.plotting(True)


plt.show()



