import cv2
from matplotlib import pyplot as plt
import numpy as np
def save_img(path, img):
    cv2.imwrite(path, img)
    print(path, "is saved!")


def display_img(img):
    cv2.imshow('Result', img)
    cv2.waitKey()


def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # if need double type, uncomment the following
    # out = image.astype(float)
    return image


def read_colorimg(path):
    image = cv2.imread(path)
    # if need double type, uncomment the following
    # out = image.astype(float)
    return image

def save_fig(path,img):
    fig, axs = plt.subplots(1,1,figsize=(6, 2),dpi=400)
    
    pos=axs.imshow(img)
    # fig.colorbar(pos, ax=axs)
    plt.axis('off')
    plt.autoscale(tight=True)
    fig.savefig(path,pad_inches=0,bbox_inches='tight')
    print(path, "is saved!")


def save_as_image(path,img_flat):
    # adapted from https://gist.github.com/fzliu/64821d31816bce595a4bbd98588b37f5
    img_flat = img_flat[0]
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    cv2.imwrite(path, img)
    print(path, "is saved!")

    # fig, axs = plt.subplots(1,1,figsize=(6, 2),dpi=400)
    
    # pos=axs.imshow(img)
    # # fig.colorbar(pos, ax=axs)
    # plt.axis('off')
    # plt.autoscale(tight=True)
    # fig.savefig(path,pad_inches=0,bbox_inches='tight')
    # print(path, "is saved!")
