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
from PIL import Image
from PIL import ImageFile
import random 
from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
import skimage
from skimage import measure
import numpy as np
import pandas as pd
import os
import pydicom
import scipy.ndimage
from skimage.transform import resize,downscale_local_mean
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matrixprofile as mp
from matrixprofile.matrixProfile import stomp
from matrixprofile.motifs import motifs
from matrixprofile.discords import discords
import numpy.fft as fft
from numpy.fft  import fft2, ifft2
import cv2
np.seterr(divide='ignore', invalid='ignore');

### define work directory
mypath = "/home/qian/Desktop/projects/iMP/COVID-CT"

batchsize=10
s = 2 
w = 4
h = 4


# define torch normalizer
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transfer image to tensor
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    normalize
])

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

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



class CovidCTDataset():
    # data read and preparation
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID,stride,w,h,transform=None):
        self.root_dir = root_dir
        self.s = stride
        self.w = w
        self.h = h
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.img_list[idx][0]
        imageN = Image.open(img_path).convert('RGB').resize((256, 256))        
        image = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        P = split_lung_parenchyma(image,256*256,5)
        P_resized = cv2.resize(P, (32, 32))
        mp = np.full((32-w+1,32-h+1), np.inf)
        for (x1, y1, window1) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
            for (x2, y2, window2) in sliding_window(P_resized, stepSize=s, windowSize=(w, h)):
                if (x2 not in range(x1-w+1,x1+w+1)) and (y2 not in range(y1-h+1,y1+h+1)):
                    dist = calculateDistance(window1, window2)
                    if dist < mp[x1,y1]:
                        mp[x1,y1] = dist 
        mp = cv2.resize(np.transpose(mp),(256,256))
        WP = P + mp*P
        WP = np.interp(WP, (WP.min(), WP.max()), (0, 255))
        WP = Image.fromarray(np.uint8(WP) , 'L')
        WP = WP.resize((256,256))
        WPN = Image.new("RGB", WP.size)
        WPN.paste(WP)

        if self.transform:
            WPN = self.transform(WPN)        
            imageN = self.transform(imageN) 
        
        sample = {'img': WPN,
                  'mp': mp,
                  'lung':P,
                  'image': imageN,
                  'label': int(self.img_list[idx][1])}
        return sample


if __name__ == '__main__':
    trainset = CovidCTDataset(root_dir='{0}/Images-processed'.format(mypath),
                              txt_COVID='{0}/Data-split/COVID/trainCT_COVID.txt'.format(mypath),
                              txt_NonCOVID='{0}/Data-split/NonCOVID/trainCT_NonCOVID.txt'.format(mypath),
                              transform= transformer,stride=s,w=w,h=h)
    valset = CovidCTDataset(root_dir='{0}/Images-processed'.format(mypath),
                              txt_COVID='{0}/Data-split/COVID/valCT_COVID.txt'.format(mypath),
                              txt_NonCOVID='{0}/Data-split/NonCOVID/valCT_NonCOVID.txt'.format(mypath),
                              transform= transformer,stride=s,w=w,h=h)
    testset = CovidCTDataset(root_dir='{0}/Images-processed'.format(mypath),
                              txt_COVID='{0}/Data-split/COVID/testCT_COVID.txt'.format(mypath),
                              txt_NonCOVID='{0}/Data-split/NonCOVID/testCT_NonCOVID.txt'.format(mypath),
                              transform= transformer,stride=s,w=w,h=h)

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    




device = 'cuda'
def train(optimizer, epoch):    
    model.train()   
    train_loss = 0
    train_correct = 0    
    for batch_index, batch_samples in enumerate(train_loader):        
        data, target = batch_samples['image'].to(device), batch_samples['label'].to(device)
        data = data[:, 0, :, :]
        data = data[:, None, :, :]    
        optimizer.zero_grad()
        output = model(data)     
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))



def train_mp(optimizer, epoch):    
    model.train()   
    train_loss = 0
    train_correct = 0    
    for batch_index, batch_samples in enumerate(train_loader):        
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        data = data[:, 0, :, :]
        data = data[:, None, :, :]    
        optimizer.zero_grad()
        output = model(data)     
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))    



def val(epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    results = []    
    TP = 0
    TN = 0
    FN = 0
    FP = 0   
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['image'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
                    
    return targetlist, scorelist, predlist


def val_mp(epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    results = []    
    TP = 0
    TN = 0
    FN = 0
    FP = 0   
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
                    
    return targetlist, scorelist, predlist


def test(epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    results = []   
    TP = 0
    TN = 0
    FN = 0
    FP = 0   
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['image'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist
    


def test_mp(epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    results = []   
    TP = 0
    TN = 0
    FN = 0
    FP = 0   
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        tpr_list = []
        fpr_list = []        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
            output = model(data)            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
           
    return targetlist, scorelist, predlist
    
    

class DenseNetModel(nn.Module):

    def __init__(self):
        super(DenseNetModel, self).__init__()

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits
    
model = DenseNetModel().cuda()
modelname = 'DenseNet'








# train_mp
bs = 10
votenum = 10

import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

total_epoch = 1000
for epoch in range(1, total_epoch+1):
    train_mp(optimizer, epoch)    
    targetlist, scorelist, predlist = val_mp(epoch)
    #print('target',targetlist)
    #print('score',scorelist)
    #print('predict',predlist)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    if epoch % votenum == 0:        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        vote_score = vote_score/votenum        
        print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
                
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUCp', roc_auc_score(targetlist, vote_pred))
        print('AUC', AUC)
         
        torch.save(model.state_dict(), "model_backup/mp_{}.pt".format(modelname))  
        
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())
        print('\n Data is mp. The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open('model_result/mp_{}.txt'.format(modelname), 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()


# test_mp
bs = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 10
for epoch in range(1, total_epoch+1):
    
    targetlist, scorelist, predlist = test_mp(epoch)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()

    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    print('precision',p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1',F1)
    print('acc',acc)
    AUC = roc_auc_score(targetlist, vote_score)
    print('AUC', AUC)

    if epoch % votenum == 0:
        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1

        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUC', AUC)
         
        vote_pred = np.zeros((1,testset.__len__()))
        vote_score = np.zeros(testset.__len__())
        print('vote_pred',vote_pred)
        print('\n Data is mp. The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open(f'model_result/test_mp_{modelname}.txt', 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()
        









# train
bs = 10
votenum = 10

import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

total_epoch = 1000
for epoch in range(1, total_epoch+1):
    train(optimizer, epoch)    
    targetlist, scorelist, predlist = val(epoch)
    #print('target',targetlist)
    #print('score',scorelist)
    #print('predict',predlist)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    if epoch % votenum == 0:        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1
        vote_score = vote_score/votenum        
        print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
                
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUCp', roc_auc_score(targetlist, vote_pred))
        print('AUC', AUC)
         
        torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))  
        
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open('model_result/{}.txt'.format(modelname), 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()




# test
bs = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []

vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 10
for epoch in range(1, total_epoch+1):
    
    targetlist, scorelist, predlist = test(epoch)
    vote_pred = vote_pred + predlist 
    vote_score = vote_score + scorelist 
    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()

    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    print('precision',p)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1',F1)
    print('acc',acc)
    AUC = roc_auc_score(targetlist, vote_score)
    print('AUC', AUC)

    if epoch % votenum == 0:
        
        # major vote
        vote_pred[vote_pred <= (votenum/2)] = 0
        vote_pred[vote_pred > (votenum/2)] = 1

        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, vote_score)
        print('AUC', AUC)
         
        vote_pred = np.zeros((1,testset.__len__()))
        vote_score = np.zeros(testset.__len__())
        print('vote_pred',vote_pred)
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open(f'model_result/test_{modelname}.txt', 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()
        

        