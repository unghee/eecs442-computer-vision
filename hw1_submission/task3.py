import cv2
import matplotlib.pyplot as plt


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


i1=imageplot('indoor.png','B','G','R')
i1.plotting(False)

i2=imageplot('outdoor.png','B','G','R')
i2.plotting(False)

i2=imageplot('indoor.png','L','A','B')
i2.plotting(True)
i2=imageplot('outdoor.png','L','A','B')
i2.plotting(True)


plt.show()


# image = cv2.imread('indoor.png')
# b,g,r = cv2.split(image)

# lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
# blab,glab,rlab = cv2.split(lab)


# fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

# axs[0].imshow(b, cmap='gray')
# axs[1].imshow(g, cmap='gray')
# axs[2].imshow(r, cmap='gray')

# axs[0].set_title('B')
# axs[1].set_title('G')
# axs[2].set_title('R')

# fig.suptitle('RGB Color Space')
# # plt.show()


# fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

# axs2[0].imshow(blab, cmap='gray')
# axs2[1].imshow(glab, cmap='gray')
# axs2[2].imshow(rlab, cmap='gray')

# axs2[0].set_title('L')
# axs2[1].set_title('A')
# axs2[2].set_title('B')

# fig2.suptitle('LAB Color Space')
# plt.show()

