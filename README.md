# eecs442-computer-vision
[EECS 442  University of Michigan Computer Vision Course](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_F19/index.html#syllabus)

# [HW1](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_F19/resources/HW1.pdf)
## 1.  Camera projection Matrix 
plot the resulting view of the cube from rotation transformations  
Implement an orthographic camera 
<img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/problem1_c.png" width ="500">
<img src="https://github.com/unghee/eecs442-computer-vision/blob/master/hw1_submission/images/ortho.png" width ="500">
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
