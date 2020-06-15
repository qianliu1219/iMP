import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import hickle as hkl
from torch.autograd import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from shutil import copyfile
from torchvision.datasets import ImageFolder
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor
import torchxrayvision as xrv
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from PIL import ImageFile
import random 
from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
import skimage
import numpy as np
import pandas as pd
import os
import pydicom
import scipy.ndimage
from skimage.transform import resize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matrixprofile as mp
from matrixprofile.matrixProfile import stomp
from matrixprofile.motifs import motifs
from matrixprofile.discords import discords
import numpy.fft as fft
from numpy.fft  import fft2, ifft2
from itertools import chain
import cv2 
from skimage import measure
np.seterr(divide='ignore', invalid='ignore');


s = 1 
w = 8
h = 8






def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-windowSize[0]+1, stepSize):
        for x in range(0, image.shape[1]-windowSize[1]+1, stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def calculateDistance(i1, i2):
    # calculate Euclidean distance
    return np.sum((i1-i2)**2)

def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)
    height, width = img.shape
    img_exp=np.zeros((height+20,width+20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10]=copyimg
    mask1 = np.zeros([height+22, width+22],np.uint8)  
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1)
    cv2.floodFill(np.float32(img_exp), mask2, (width_exp-1, height_exp-1), 1)
    cv2.floodFill(np.float32(img_exp), mask3, (width_exp-1, 0), 1)
    cv2.floodFill(np.float32(img_exp), mask4, (0, height_exp-1), 1)
    mask = mask1 | mask2 | mask3 | mask4
    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output
 
def split_lung_parenchyma(img,size,thr):
    try:
        img_thr= cv2.adaptiveThreshold(img, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV,size,thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV,999,thr).astype(np.uint8)
    img_thr=255-img_thr
    img_test=measure.label(img_thr, connectivity = 1)
    props = measure.regionprops(img_test)
    areas=[prop.area for prop in props]
    ind_max_area=np.argmax(areas)+1
    del_array = np.zeros(img_test.max()+1)
    del_array[ind_max_area]=1
    del_mask=del_array[img_test]
    img_new = img_thr*del_mask
    mask_fill=fill_water(img_new)
    img_new[mask_fill==1]=255
    img_out=img*~img_new.astype(bool)
    return img_out



nCT_path = "/home/qian/Desktop/projects/iMP/COVID-CT/Images-processed/CT_NonCOVID"        
img_list = os.listdir(nCT_path)
risk_score_nCT = []
for c in img_list:
    img_path = os.path.join(nCT_path,c)
    image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    P = split_lung_parenchyma(image,256*256,-5)
    P_resized = cv2.resize(P, (32, 32))
    mp = np.full((32-w+1,32-h+1), np.inf)
    for (x1, y1, window1) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
        for (x2, y2, window2) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
            if (x2 not in range(x1-w+1,x1+w+1)) and (y2 not in range(y1-h+1,y1+h+1)):
                dist = calculateDistance(window1, window2)
                if dist < mp[x1,y1]:
                    mp[x1,y1] = dist 
    #mp = cv2.resize(np.transpose(mp),(256,256))
    risk_score = mp.sum()
    print(risk_score)
    risk_score_nCT.append(risk_score) 


with open("risk_score_nCT.txt", 'w') as f:
    for rs in risk_score_nCT:
        f.write('{}\n'.format(rs))



pCT_path = "/home/qian/Desktop/projects/iMP/COVID-CT/Images-processed/CT_COVID"        
img_list = os.listdir(pCT_path)
risk_score_pCT = []
for c in img_list:
    img_path = os.path.join(pCT_path,c)
    image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    P = split_lung_parenchyma(image,256*256,-5)
    P_resized = cv2.resize(P, (32, 32))
    mp = np.full((32-w+1,32-h+1), np.inf)
    for (x1, y1, window1) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
        for (x2, y2, window2) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
            if (x2 not in range(x1-w+1,x1+w+1)) and (y2 not in range(y1-h+1,y1+h+1)):
                dist = calculateDistance(window1, window2)
                if dist < mp[x1,y1]:
                    mp[x1,y1] = dist 
    #mp = cv2.resize(np.transpose(mp),(256,256))
    risk_score = mp.sum()
    print(risk_score)
    risk_score_pCT.append(risk_score) 


with open("risk_score_pCT.txt", 'w') as f:
    for rs in risk_score_pCT:
        f.write('{}\n'.format(rs))
