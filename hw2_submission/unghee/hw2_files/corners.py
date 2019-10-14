import os
from common import read_img, save_img 
import matplotlib.pyplot as plt
import numpy as np 

def corner_score(image, u=5, v=5, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image. 

    output = None # implement     

    return output

def harris_detector(image, window_size=(5,5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    # 
    # You can use same-padding for intensity (or zero-padding for derivatives) 
    # to handle window values outside of the image. 

    ## compute the derivatives 
    Ix = None
    Iy = None 

    Ixx = None
    Iyy = None
    Ixy = None

    # For each location of the image, construct the structure tensor and calculate the Harris response
    response = None

    return response

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Feature Detection #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # define offsets and window size and calulcate corner score
    u, v, W = 0, 2, (5,5)
    
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")

if __name__ == "__main__":
    main()
