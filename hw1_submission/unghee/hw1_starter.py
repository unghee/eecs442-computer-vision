# This starter code requires functions in the Dolly Zoom Notebook to work
from dolly_zoom import *

import os
import imageio

# Call this function to generate gif. make sure you have rotY() implemented.


def rotY(theta):
    result = np.array([[np.cos(theta),0,np.sin(theta)],
        [0,1,0],
        [-np.sin(theta),0,np.cos(theta)]])

    return result

def rotX(theta):
    result = np.array([[1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]])

    return result

def rotZ(theta):
    result = np.array([[np.cos(theta),-np.sin(theta),0],
        [np.sin(theta),np.cos(theta),0],
        [0,0,1]])

    return result

def generate_gif():
    n_frames = 30
    if not os.path.isdir("frames"):
        os.mkdir("frames")
    fstr = "frames/%d.png"
    for i,theta in enumerate(np.arange(0,2*np.pi,2*np.pi/n_frames)):
        fname = fstr % i
        renderCube(f=15, t=(0,0,3), R=rotY(theta))
        plt.savefig(fname)
        plt.close()

    with imageio.get_writer("cube.gif", mode='I') as writer:
        for i in range(n_frames):
            frame = plt.imread(fstr % i)
            writer.append_data(frame)
            os.remove(fstr%i)
            
    os.rmdir("frames")
