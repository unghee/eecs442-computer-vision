import cv2
from matplotlib import pyplot as plt

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
