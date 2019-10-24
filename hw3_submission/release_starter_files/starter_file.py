import numpy as np
from matplotlib import pyplot as plt
from common import *
# feel free to include libraries needed
import cv2

def homography_transform(X, H):
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


def fit_homography(XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix

    A =  np.empty((0,9), int)
    for i in range(len(XY)):

        XN = np.concatenate((XY[i,0:2],[1]))
        y_prime = XY[i,3]
        x_prime = XY[i,2]

        line1 = np.concatenate((XN, np.zeros(3),-x_prime*XN))
        line2 = np.concatenate((np.zeros(3),-XN,y_prime*XN))

        lines = np.vstack((line1,line2))
        A = np.vstack((A,lines))

    eigenValues, eigenVectors = np.linalg.eig(np.matmul(A.T,A))
  
    U, S, VH= np.linalg.svd(A, full_matrices=True)

    # H = eigenVectors[np.argmin(eigenValues)].reshape((3,3))

    # H = H/np.linalg.norm(H)
    H = VH[len(S)-1]
    H = H.reshape((3,3))


    # H = None
    return H


def p1():
    # 1.2.3 - 1.2.5
    # TODO
    # 1. load points X from p1/transform.npy

    # 2. fit a transformation y=Sx+t

    # 3. transform the points 

    # 4. plot the original points and transformed points
    points=np.load('p1/transform.npy')
    X = points[:,0:2]; Y = points[:,2:4] 

    A = np.hstack([X, np.ones((len(X),1))])

    W= np.linalg.lstsq(A,Y)[0]

    S_ = W[:2,:];t_ = W[2,:]

    Y_est=np.matmul(A,W)

    summed_error = sum((Y_est-Y)**2)
    print('summed error', summed_error)

    fig, ax = plt.subplots()
    plt.scatter(X,Y,c="green", label = 'ground truth') #Y
    plt.scatter(X,Y_est,c="blue", label ='estimated') #Y
    ax.legend()
    plt.show()

    # 1.2.6 - 1.2.8
    case = 8 # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load('p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography() 
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:,:2], H)
        # 4. Visualize points as three images in one figure
        # the following codes plot figure for you
        plt.scatter(XY[:,1],XY[:,0],c="red") #X
        plt.scatter(XY[:,3],XY[:,2],c="green") #Y
        plt.scatter(Y_H[:,1],Y_H[:,0],c="blue") #Y_hat
        plt.savefig('./case_'+str(i))
        plt.close()



def stitchimage(imgleft, imgright):
    # TODO
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgleft,None)
    kp2, des2 = sift.detectAndCompute(imgright,None)
    imgleft_surf = cv2.drawKeypoints(imgleft,kp1,None,(255,0,0),4)
    imgright_surf = cv2.drawKeypoints(imgright,kp2,None,(255,0,0),4)
    save_img('uttower_left_grey_surf.jpg',imgleft_surf)
    save_img('uttower_right_grey_surf.jpg',imgright_surf)

    des1 = (des1 - np.mean(des1))/des1.std(axis=0)
    des2 = (des2 - np.mean(des2))/des2.std(axis=0)

    pairs = []
    matches =[]

    for i in range(len(des2)):
        oldDist =10000
        for j in range(len(des1)):
            dist = np.linalg.norm(des1[j]-des2[i])
            if dist < oldDist:
                secondsmallest = oldDist
                oldDist = dist
                pairs_smallest=[j,i]
                # get the second smallest and do the ratio test
            else: 
                if dist < secondsmallest:
                    secondsmallest = dist

        if oldDist/secondsmallest < 0.75:
            matches.append(cv2.DMatch(pairs_smallest[0],pairs_smallest[1],oldDist))


    matches = sorted(matches, key = lambda x:x.distance)

    # 2. select paired descriptors

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers

    XY_query = np.asarray([kp1[matches[i].queryIdx].pt for i in range(len(matches))])
    XY_train = np.asarray([kp2[matches[i].trainIdx].pt for i in range(len(matches))])

    sampled_stack = np.zeros((1,4))
    numTrials = 50

    bestLine, bestCount = None, -1 

    inliers_matches =[]
    for trial in range(numTrials):
        rand_matches=np.random.choice(matches,5)

        for i in range(len(rand_matches)):
            kps=[kp1[rand_matches[i].queryIdx].pt,kp2[rand_matches[i].trainIdx].pt]
            kps = np.asarray(kps)
            kps = np.concatenate(kps)
            sampled_stack=np.vstack((sampled_stack,kps))

        sampled_stack = sampled_stack[1:]
        
        H=fit_homography(sampled_stack)

        XY_H = homography_transform(XY_query, H)
        XY_H = XY_H[:,:2]
        E = np.linalg.norm(XY_train-XY_H,axis=1)
        inliers= XY_train[E<30]
        matches = np.asarray(matches)
        matches_inliers = matches[E<30]

        numb_inliers = len(inliers)
        if numb_inliers > bestCount:
            residual =np.sum(E*E)/bestCount
            bestLine, bestCount, best_inliers = H, numb_inliers, matches_inliers


    img3 = cv2.drawMatches(imgleft_surf ,kp1,imgright_surf,kp2,best_inliers, None, flags=2)
    plt.imshow(img3),plt.show()
    save_img('matches.jpg',img3)

    print('numb_inliers',bestCount,'residual',residual)

    # 4. warp one image by your transformation 
    #    matrix
    #
    #    Hint: 
    #    a. you can use opencv to warp image
    #    b. Be careful about final image size

    # t = [0,0]
    # Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])

    height, width, channel = imgright.shape

    # im1Reg = cv2.warpPerspective(imgleft, Ht.dot(bestLine), (width+t[0], height))
    im1Reg = cv2.warpPerspective(imgleft, bestLine, (width, height))

    im1Reg = im1Reg.astype(np.uint8)
    # im1Reg= cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)


    plt.imshow(im1Reg),plt.show()
    save_img('warped.jpg',im1Reg)

    # im1Reg[t[1]:height+t[1],t[0]:width+t[0]] = imgright
    plt.imshow(im1Reg),plt.show()


    alpha = 0.5
    beta = (1.0 - alpha)
    dst = np.uint8(alpha*(im1Reg)+beta*(imgright))
    plt.imshow( dst), plt.show()

    # 5. combine two images, use average of them
    #    in the overlap area

    return bestLine, dst


def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_colorimg(p1)
    imgright = read_colorimg(p2)

    imgleft = imgleft.astype(np.uint8)
    imgright = imgright.astype(np.uint8)

    # imgleft= cv2.cvtColor(imgleft, cv2.COLOR_BGR2RGB)
    # imgright= cv2.cvtColor(imgright, cv2.COLOR_BGR2RGB)

    # imgleft= cv2.cvtColor(imgleft, cv2.COLOR_BGR2GRAY)
    # imgright= cv2.cvtColor(imgright, cv2.COLOR_BGR2GRAY)

    save_img('left.jpg',imgleft)
    save_img('right.jpg',imgright)

    # stitch image
    H, output = stitchimage(imgleft, imgright)

    # save stitched image
    save_img( savename + '.jpg', output)


def transform_img(imgbase,img,mat):

    img_shape = np.shape(img)

    img_out= np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            xy = [[i],[j],[1]]
            xy_scaled = np.matmul(mat,xy) 

            xy_scaled=xy_scaled/xy_scaled[2][0]
            if int(xy_scaled[0][0])<img_shape[0] and int(xy_scaled[1][0])<img_shape[1]:
                img_out[int(xy_scaled[0][0]),int(xy_scaled[1][0])] = img[i,j]

    return img_out

if __name__ == "__main__":
    # Problem 1
    # p1();

    # Problem 2
    # p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    # p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

    # Problem 3
    # TODO
    # add your code for implementing Problem 3
    # 
    # Hint:
    # you can use functions in Problem 2 here

    scale_matrix = np.array([[0.3,0,180],
                            [0,0.2,370],
                            [0,0,1]])
    imgleft = read_colorimg('bbb_front.png')
    imgright = read_colorimg('M.png')
    imgside= read_colorimg('bbb_side.png')

    imgleft= cv2.cvtColor(imgleft, cv2.COLOR_BGR2RGB)
    imgright= cv2.cvtColor(imgright, cv2.COLOR_BGR2RGB)
    imgside= cv2.cvtColor(imgside, cv2.COLOR_BGR2RGB)

    imgleft = imgleft.astype(np.uint8)
    imgright = imgright.astype(np.uint8)
    imgside = imgside.astype(np.uint8)

    img_shape = np.shape(imgright)
    imgleft = imgleft[:-2,:,:]


    img_ov = transform_img(imgleft,imgright,scale_matrix)


    img_ov = img_ov.astype(np.uint8)
    alpha = 0.7
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(imgleft, alpha, img_ov, beta, 0.0)
    dst = np.uint8(alpha*(imgleft)+beta*(img_ov))
 
    save_fig('./' + 'bbb_front_ov' + '.jpg', dst)


    H=stitchimage(imgleft,imgside)
    print('H',H)

    height, width, channels = img_ov.shape
    imrightReg = cv2.warpPerspective(img_ov, H, (width, height))
    plt.imshow(imrightReg ),plt.show()

    imgside = imgside[:,:-2,:]
    imrightReg = imrightReg[:-2,:,:]

    dst2 = cv2.addWeighted(imgside, alpha, imrightReg , beta, 0.0)
    dst2 = np.uint8(alpha*(imgside)+beta*(imrightReg ))

    plt.imshow(dst2 ),plt.show()

    save_fig('./' + 'bbb_side_ov' + '.jpg', dst2)



