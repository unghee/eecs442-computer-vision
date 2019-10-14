import numpy as np
import os
from common import *
import math


## Image Patches ##
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    shape_image=np.shape(image)
    sliced_list =[]

    for i in range(int(shape_image[0]/16)):
        for j in range(int(shape_image[1]/16)):
            image_iter=image[16*i:16*(i+1),16*j:16*(j+1)]

            sliced_list.append((image_iter-np.mean(image_iter))/np.std(image_iter))

    output = sliced_list

    return output


## Gaussian Filter ##
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W
    image = image*1.0 # change to float
    shape_image = np.shape(image)
    shape_kernel = np.shape(kernel)
    padded_image = np.zeros((shape_image[0]+shape_kernel[0]-1,shape_image[1]+shape_kernel[1]-1))
    padded_image_shape = np.shape(padded_image)
    # padded_image[1:-1,1:-1] = image
    # pad = lambda s: None if s is 1 else int(s/2)
    pad = lambda s: int(s/2)
    if pad(shape_kernel[0]) == 0 and pad(shape_kernel[1]) != 0:
        padded_image[:,pad(shape_kernel[1]):-pad(shape_kernel[1])] = image

    elif pad(shape_kernel[0]) != 0  and pad(shape_kernel[1]) ==0 :
        padded_image[pad(shape_kernel[0]):-pad(shape_kernel[0]),:] = image

    elif pad(shape_kernel[0]) != 0  and pad(shape_kernel[1]) != 0 :
        padded_image[pad(shape_kernel[0]):-pad(shape_kernel[0]),pad(shape_kernel[1]):-pad(shape_kernel[1])] = image

    else:
        padded_image = image



   
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros(shape_image)


    for x in range(shape_image[1]):
        for y in range(shape_image[0]):
            output[y][x] = np.sum(padded_image[y:y+shape_kernel[0],x:x+shape_kernel[1]]*kernel)


    return output


## Edge Detection ##
def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
    kx = np.array([[-1, 0, 1]]) # 1 x 3
    kx = 0.5*kx
    ky = np.transpose(kx)  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2+Iy**2)

    return grad_magnitude, Ix, Iy


## Sobel Operator ##
def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    kx_gauss = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # kx_gauss = kx_gauss/np.sum(abs(kx_gauss))
    ky_gauss = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # ky_gauss = ky_gauss/np.sum(abs(ky_gauss))

    Gx = convolve(image,kx_gauss)
    Gy = convolve(image,ky_gauss)

    grad_magnitude = np.sqrt(Gx**2+Gy**2)

    # Gx, Gy, grad_magnitude = convolve(image,kx_gauss), convolve(image,kx_gauss), None

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    # You are encouraged not to use sobel_operator() in this function.

    # TODO: Use convolve() to complete the function
    output = [None]*len(angles)
    kx_gauss = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky_gauss = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    for i in range(len(angles)):
        K_alpha = kx_gauss*np.cos(angles[i])+ky_gauss*np.sin(angles[i])
        output[i] = convolve(image,K_alpha)

    return output


def gauss(m,n,sig):
    ker=np.zeros((m,n))

    for i in range(ker.shape[1]):
        for j in range(ker.shape[0]):
            ker[j][i] = 1/(2*math.pi*sig**2)*np.exp(-(i**2+j**2)/(2*sig**2))

    ker = ker/np.sum(ker)

    return ker


def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patches = image_patches(img)
    # TODO choose a few patches and save them
    chosen_patches = patches[5]
    chosen_patches = chosen_patches.astype(np.uint8)

    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.  There is tolerance for the kernel.
    # kernel_gaussian = None
    kernel_gaussian = gauss(3,3,np.sqrt(1/(2*math.log(2))))
    # kernel_gaussian = gauss(3,3,10)

    filtered_gaussian = convolve(img, kernel_gaussian)
    filtered_gaussian = filtered_gaussian.astype(np.uint8)
    # display_img(filtered_gaussian)

    fig, axs = plt.subplots(1,1,figsize=(9, 3),dpi=350)
    axs.imshow(filtered_gaussian, cmap='gray')
    fig.savefig('./gaussian_filter/q2_gaussian.png')
    # save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    edge_detect = edge_detect.astype(np.uint8)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    edge_with_gaussian = edge_with_gaussian.astype(np.uint8)
    # display_img(edge_with_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    # Gx, Gy, edge_sobel = Gx.astype(np.uint8),Gy.astype(np.uint8),edge_sobel.astype(np.uint8)

    # save_img(Gx, "./sobel_operator/q2_Gx.png")
    # save_img(Gy, "./sobel_operator/q2_Gy.png")
    # save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    save_fig(Gx,"./sobel_operator/q2_Gx.png")
    save_fig(Gy,"./sobel_operator/q2_Gy.png")
    save_fig(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        # steerable = steerable.astype(np.uint8)
        # save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))
        save_fig(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    # save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_fig(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")
    save_fig(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code

    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()
