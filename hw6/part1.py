import numpy as np

from matplotlib import pyplot as plt
def projection_transform(X, H):
    # TODO
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)

    X_h=np.hstack((X,np.ones((np.size(X,0),1))))


    Y = np.matmul(H,X_h.T)
    Y = Y.T
    K = Y[:,2]
    K = K[:,None]
    Y = Y/K
    return Y


def fit_projection(UV,XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix

    A =  np.empty((0,12), int)
    for i in range(len(XY)):

        v = UV[i,1]
        u = UV[i,0]
        X = XYZ[i]
        X = np.append(X,1)

        # line1 = np.concatenate((XN, np.zeros(3),-x_prime*XN))
        # line2 = np.concatenate((np.zeros(3),-XN,y_prime*XN))

        line1 = np.concatenate((np.zeros(4), -X, v*X))
        line2 = np.concatenate((X, np.zeros(4),-u*X))
        # line3 = np.concatenate((-v*X, -u*X,np.zeros(4)))

        # lines = np.vstack((line1,line2,line3))
        lines = np.vstack((line1,line2))
        A = np.vstack((A,lines))

    eigenValues, eigenVectors = np.linalg.eig(np.matmul(A.T,A))
  
    U, S, VH= np.linalg.svd(A, full_matrices=True)

    # H = eigenVectors[np.argmin(eigenValues)].reshape((3,3))
    # H = H/np.linalg.norm(H)

    H = VH[len(S)-1]
    H = H.reshape((3,4))

    return H

UV=np.loadtxt('data/pts2d-norm-pic.txt')
XYZ=np.loadtxt('data/pts3d-norm.txt')

H = fit_projection(UV,XYZ)
print(H)

Y_H = projection_transform(XYZ, H)
fig, ax = plt.subplots()
# plt.scatter(XYZ[:,1],XYZ[:,0],c="red") #X
plt.scatter(UV[:,0],UV[:,1],c="green", label = 'ground truth') #Y
plt.scatter(Y_H[:,0],Y_H[:,1],c="blue", label = 'estimated') #Y_hat
ax.legend()
plt.show()


# pts2[:,0]= u
# pts2[:,1] = v

# pts

# P = np.matmul(pts2, np.linalg.pinv(pts3))