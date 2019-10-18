from common import * 
import matplotlib.pyplot as plt
import numpy as np 
from scipy import signal
import math

def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation 
    # Input-    image: image of size HxW
    #           sigma: scalar standard deviation of Gaussian Kernel
    # Output-   Gaussian filtered image of size HxW
    image = image*1.0
    H, W = image.shape
    # -- good heuristic way of setting kernel size 
    kernel_size = int(2 * np.ceil(2*sigma) + 1)

    # make sure that kernel size isn't too big and is odd 
    kernel_size = min(kernel_size, min(H,W)//2)     
    if kernel_size % 2 == 0: kernel_size = kernel_size + 1  

    #TODO implement gaussian filtering with size kernel_size x kernel_size 
    # feel free to use your implemented convolution function or a convolution function from a library 
    kernel = np.zeros((kernel_size,kernel_size))
    for i in range(-int(kernel_size/2),int(kernel_size/2)):
    	for j in range(-int(kernel_size/2),int(kernel_size/2)):
    		kernel[j][i] = (1/(2*math.pi*sigma*sigma))*np.exp(-(i**2+j**2)/(2*sigma*sigma))

    kernel = kernel*1.0
    # kernel = kernel/np.sum(kernel)
    output = signal.convolve2d(image, kernel, boundary='symm', mode='same')	


    return output 

def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input-    image: image of size HxW
    #           min_sigma: smallest sigma in scale space
    #           k: scalar multiplier for scale space
    #           S: number of scales considers
    # Output-   Scale Space of size HxWx(S-1)

    image_shape = np.shape(image)
    output = np.zeros((image_shape[0],image_shape[1],S-1))

    for i in range(S-1):
    	gauss_2 = gaussian_filter(image,min_sigma*k**(i+1))
    	gauss_1 = gaussian_filter(image,min_sigma*k**(i))

    	output[:,:,i] = gauss_2-gauss_1

    return output


##### You shouldn't need to edit the following 3 functions 
def find_maxima(scale_space, k_xy=5, k_s=1):
    # Extract the peak x,y locations from scale space
    # Input-    scale_space: Scale space of size HxWxS
    #           k: neighborhood in x and y 
    #           ks: neighborhood in scale
    # Output-   list of (x,y) tuples; x<W and y<H
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 

    H,W,S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i-k_xy):min(i+k_xy,H), 
                                        max(0, j-k_xy):min(j+k_xy,W), 
                                        max(0, s-k_s) :min(s+k_s,S)]
                mid_pixel = scale_space[i,j,s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel is larger than all the neighbors; append maxima 
                if np.sum(mid_pixel > neighbors) == num_neighbors:
                    maxima.append( (i,j,s) )
    return maxima

def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    # Visualizes the scale space
    # Input-    scale_space: scale space of size HxWxS
    #           min_sigma: the minimum sigma used 
    #           k: the sigma multiplier 
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S))) 
    p_w = int(np.ceil(S/p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i+1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i, min_sigma * k**(i+1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig 
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    

def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    # Visualizes the maxima on a given image
    # Input-    image: image of size HxW
    #           maxima: list of (x,y) tuples; x<W, y<H
    #           file_path: path to save image. if None, display to screen
    # Output-   None 
    H, W = image.shape
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for maximum in maxima:
        y,x,s= maximum 
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2 * min_sigma * (k ** s))
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    


def main():
    image = read_img('polka.png')
    image = image*1.0

    ### -- Detecting Polka Dots -- ## 
    print("Detect small polka dots")
    # -- Detect Small Circles
    sigma_1, sigma_2 = 8, 12
    gauss_1 = gaussian_filter(image,sigma_1)
    gauss_2 = gaussian_filter(image,sigma_2)

    # calculate difference of gaussians
    DoG_small = gauss_2-gauss_1

    # visualize maxima 
    maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    visualize_scale_space(DoG_small, sigma_1, sigma_2/sigma_1,'polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, 'polka_small.png')
    
    # -- Detect Large Circles
    print("Detect large polka dots")
    # sigma_1, sigma_2 = 45, 50
    sigma_1, sigma_2 = 55, 58
    gauss_1 = gaussian_filter(image,sigma_1)
    gauss_2 = gaussian_filter(image,sigma_2)

    # calculate difference of gaussians 
    DoG_large = gauss_2 - gauss_1
    
    # visualize maxima 
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2/sigma_1, 'polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, 'polka_large.png')


    ## -- TODO Implement scale_space() and try to find both polka dots 

    min_sig = 6
    k = 1.44
    spaces=scale_space(image,min_sig,k,8)
    maxima = find_maxima(spaces, k_xy=13,k_s=3)
    visualize_scale_space(spaces, min_sig, k, 'polka_scale_space.png')
    visualize_maxima(image, maxima, min_sig, k, 'polka_small2large.png')

    ## -- TODO Try to find the cells in any of the cell images in vgg_cells 

    image = read_img('cells/001cell.png')

    image = -1.0*image
    min_sig = 2.3
    k = 1.12
    spaces=scale_space(image,min_sig,k,8)
    maxima = find_maxima(spaces, k_xy=17,k_s=5)
    visualize_scale_space(spaces, min_sig, k, 'cell_scale_space.png')
    visualize_maxima(image, maxima, min_sig, k, 'cell_number.png')
    print(len(maxima))


if __name__ == '__main__':
    main()
