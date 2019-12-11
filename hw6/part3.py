import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils import *
from mpl_toolkits.mplot3d import Axes3D


data=np.load('data/temple.npz')

# for key in data:
# 	print(key)

K1= data['K1']
K2= data['K2']

UV=data['pts1']
UV_prime = data['pts2']

F=cv2.findFundamentalMat(UV,UV_prime,method=cv2.FM_8POINT)[0]

E = np.dot(np.dot(K2.T,F),K1)

print(E)
R1,R2,t = cv2.decomposeEssentialMat(E)

#R1 -t

IO=np.hstack((np.eye(3), np.zeros((3,1))))
Rt = np.hstack((R2,-t))

P1 = np.dot(K1,IO)
P2 = np.dot(K2,Rt)
print(P1)
print(P2)
tri_points=cv2.triangulatePoints(P1,P2,UV.T.astype(float),UV_prime.T.astype(float))


tri_points_tran=tri_points.T
tri3 = tri_points_tran[:,3]
tri3 = tri3[:,None]
tri_tran_norm=tri_points_tran/tri3

tri_tran_norm = tri_tran_norm[:,:3]

visualize_pcd(tri_tran_norm)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(tri_tran_norm[0,:], tri_tran_norm[1,:], tri_tran_norm[2,:], c='r', marker='o')
