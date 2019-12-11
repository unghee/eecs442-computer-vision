import numpy as np

from matplotlib import pyplot as plt

import cv2

from utils import *
data=np.load('data/temple.npz')

# for key in data:
# 	print(key)

UV=data['pts1']
UV_prime = data['pts2']

# how should I get A and f? 


def fit_projection(UV,UV_prime):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix

    UV=np.hstack((UV,np.ones((np.size(UV,0),1))))
    UV_prime = np.hstack((UV_prime,np.ones((np.size(UV_prime,0),1))))

    T = [[1/np.max(UV[:,0]), 0, 0 ],
    						[0, 1/np.max(UV[:,1]), 0],
    						[0, 0, 1]]
    T = np.array(T)

    T_prime= [[1/np.max(UV_prime[:,0]), 0, 0 ],
    						[0, 1/np.max(UV_prime[:,1]), 0],
    						[0, 0, 1]]
    T_prime = np.array(T_prime)

    UVnorm = np.matmul(UV,T)
    UV_primenorm = np.matmul(UV_prime,T_prime)



    A =  np.empty((0,9), float)
    for i in range(len(UV)):

    	u = UVnorm[i,0]
    	v = UVnorm[i,1]
    	uprime = UV_primenorm[i,0]
    	vprime = UV_primenorm[i,1]

    	line1 = np.array([u*uprime, u*vprime, u, v*uprime, v*vprime,v,uprime,vprime,1])
    	A = np.vstack((A,line1))

    eigenValues, eigenVectors = np.linalg.eig(np.dot(A.T,A))
    F = eigenVectors[:,np.argmin(eigenValues)]

    # U_, S_, VH= np.linalg.svd(A, full_matrices=True)
    # F= VH[np.argmin(S_**2)]
    F = F.reshape((3,3))

    U, S, V=np.linalg.svd(F)

    S[2] = 0

    F = np.dot(U*S,V)

    F = np.dot(np.dot(T_prime.T,F),T)

    return F


F=fit_projection(UV,UV_prime)


# normalize F
# K = F[:,2]
# K = K[:,None]
F = F/F[2,2]
F_true=cv2.findFundamentalMat(UV,UV_prime,method=cv2.FM_8POINT)[0]

print(F,'my implementation')
print(F_true,'cv2 function')

image1 = cv2.imread('data/im1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('data/im2.png', cv2.IMREAD_GRAYSCALE)
draw_epipolar(image1,image2,F,UV,UV_prime)

