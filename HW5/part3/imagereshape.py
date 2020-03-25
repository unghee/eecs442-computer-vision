
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset


img = cv2.imread('/home/unghee/codes/eecs442-computer-vision/HW5/part3/starter_set/test_dev/eecs442_0114.jpg')

img = img[:256,:256,:]
cv2.imwrite('eecs442_0114_reshaped.jpg',img)