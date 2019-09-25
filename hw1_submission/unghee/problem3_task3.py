import cv2
import matplotlib.pyplot as plt
from skimage import io

image = cv2.imread('cool.jpg')
h,w,d = image.shape

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb_cropped = image_rgb[(h-w):,:,:]
shrinked_image1 = cv2.resize(image_rgb_cropped,(256,256))
# plt.imshow(image_rgb)
# plt.show()

# plt.imshow(image_rgb_cropped)
# plt.show()
plt.imshow(shrinked_image1)
plt.show()


image2= cv2.imread('warm.jpg')
h,w,d = image2.shape

image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image_rgb_cropped2 = image_rgb2[(h-w):,:,:]
shrinked_image2 = cv2.resize(image_rgb_cropped2,(256,256))
plt.imshow(shrinked_image2)
plt.show()


io.imsave('img1.jpg', shrinked_image1)
io.imsave('img2.jpg', shrinked_image2)