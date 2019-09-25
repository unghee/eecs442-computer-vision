
import os,sys

thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(thisdir)

import matplotlib.pyplot as plt
import numpy as numpy
from dolly_zoom import *
import math

import imageio
from hw1_starter import *

# (a)
# rotY, rotX implemented in hw1_starter.py
generate_gif()


#(b)
theta = math.pi/4
Rx = rotX(theta); Ry = rotY(theta)
renderCube(f=15,t=(0,0,3),R = np.matmul(Rx,Ry))
renderCube(f=15,t=(0,0,3),R = np.matmul(Ry,Rx))
plt.show()

#(c)

theta = math.acos(np.sqrt(2/3))
thetay= math.pi/4
Rx = rotX(theta); Ry = rotY(thetay)
renderCube(f=15,t=(0,0,3),R = np.matmul(Rx,Ry))
plt.show()

#(d)
renderCube(f=15,t=(0,0,3),R = np.matmul(Rx,Ry),ortho=True)
plt.show()