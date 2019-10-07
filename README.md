# eecs442-computer-vision
[EECS 442  University of Michigan Computer Vision Course](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_F19/index.html#syllabus)

# [HW1](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_F19/resources/HW1.pdf)
## 1.  Camera projection Matrix 
plot the resulting view of the cube from rotation transformations  
Implement an orthographic camera 


<img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/problem1_c.png" width ="400"><img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/ortho.png" width ="400">
## 2. Prokudin-Gorskii: Color from grayscale photographs 
Implementing the dream of Russian photographer, Sergei Mikhailovich
Prokudin-Gorskii (1863-1944), via a project invented by Russian-American vision researcher, Alexei A.
Efros (1975-present).Sergei was a visionary who believed in a future with color photography (which
we now take for granted). During his lifetime, he traveled across the Russian Empire taking photographs
through custom color filters at the whim of the czar. To capture color for the photographers of the future,
he decided to take three separate black-and-white pictures of every scene, each with a red, green, or blue
color filter in front of the camera. His hope was that you, as a student in the future, would come along and
produce beautiful color images by combining his 3 separate, filtered images.


![](https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/HW1_files/prokudin-gorskii/00125v.jpg)
![alt text](https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/00125vstacked_alinged.jpg)
![alt text](https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/HW1_files/prokudin-gorskii/00153v.jpg)
![alt text](https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/00153vstacked_alinged.jpg)

## 3. Color Spaces and illuminance

The same color may look different under different lighting conditions. Images indoor.png and outdoor.png
are two photos of a same Rubik’s cube under different illuminances  
Load the images and plot their R, G, B channels separately as grayscale images using plt.imshow()
(beware of normalization). Then convert them into LAB color space.

<p align="center">
<img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/HW1_files/outdoor.png" >
<img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/task3_1_outdoor_lab.png" >
</p>

# [HW2](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_F19/resources/HW2.pdf)
## 1. Image Filtering 
In this first section, you will explore different ways to filter images. Through these tasks you will build
up a toolkit of image filtering techniques. By the end of this problem, you should understand how the
development of image filtering techniques has led to convolution

## 2. Feature Extraction
While edges can be useful, corners are often more informative features as they are less common. In this
section, we implement a Harris Corner Detector (see: https://en.wikipedia.org/wiki/Harris Corner Detector)
to detect corners. Corners are defined as locations (x, y) in the image where a small change any direction
results in a large change in intensity if one considers a small window centered on (x, y) (or, intuitively,
one can imagine looking at the image through a tiny hole that’s centered at (x, y)). This can be contrasted
with edges where a large intensity change occurs in only one direction, or flat regions where moving in any
direction will result in small or no intensity changes. Hence, the Harris Corner Detector considers small
windows (or patches) where a small change in location leads large variation in multiple directions (hence
corner detector)

## 3. Blob Detection
One of the great benefits of computer vision is that it can greatly simplify and automate otherwise tedious
tasks. For example, in some branches of biomedical research, researchers often have to count or annotate
specific particles microscopic images such as the one seen below. Aside from being a very tedious task, this
task can be very time consuming as well as error-prone. During this course, you will learn about several
algorithms that can be used to detect, segment, or even classify cells in those settings. In this part of the
assignment, you will use the DoG filters implemented in part 1 along with a scale-space representation to
count the number of cells in a microscopy images
