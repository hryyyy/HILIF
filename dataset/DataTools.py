from torchvision.datasets.vision import VisionDataset
from PIL import Image
import cv2
import numpy as np
import os
import random
import torchvision
import io

fmax = 1 / (2 + (2 * np.sqrt(np.log(2)) / np.pi))
f_n = 5
theta_n = 8
ksize = 31
sigma_x = sigma_y = 0.5
K = 2

def rgb2gray(img):
    r,g,b = cv2.split(img)
    img_gray = 0.299 * r + 0.587 * g + 0.114 * b
    img_gray = img_gray.astype('uint8')
    return img_gray 

class Imgdataset(VisionDataset):
    def __init__(self, rootlist, process=None, transform=None, randomdrop=0):
        self.rootlist = rootlist
        self.randomdrop = randomdrop
        self.dataset = []
        self.process = process
        self.transform = transform
        for root, label in self.rootlist:  
            imglist = os.listdir(root)
            for p in imglist: 
                self.dataset.append((os.path.join(root, p), label)) 
                # print(os.path.join(root,p),os.path.join(mask,p),label)
                # print(os.path.join(root,p),os.path.join(mask,p))
            print("Loaded %s=>%d张" % (root, len(imglist)))

    def shuffle(self):
        random.shuffle(self.dataset)

    def reset(self):
        self.dataset = []
        for root, label in self.rootlist:
            imglist = os.listdir(root)
            for p in imglist:
                self.dataset.append((os.path.join(root, p), label))

    def __getitem__(self, index): 
        img, label = self.dataset[index]
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other):
        self.dataset.extend(other.dataset)
        return self

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other):
        self.dataset.extend(other.dataset)
        return self

class Imgdataset2(VisionDataset): 
    def __init__(self,rootlist,process=None,transform=None,randomdrop=0):
        self.rootlist = rootlist
        self.randomdrop = randomdrop
        self.dataset = []
        self.process = process
        self.transform = transform
        for root,label in self.rootlist:
            imglist = os.listdir(root)
            for p in imglist: 
                self.dataset.append((os.path.join(root,p),label))
                # print(os.path.join(root,p),os.path.join(mask,p),label)
                # print(os.path.join(root,p),os.path.join(mask,p))
            print("Loaded %s=>%d张" %(root,len(imglist)))

    def shuffle(self):
        random.shuffle(self.dataset)

    def reset(self):
        self.dataset = []
        for root,label in self.rootlist:
            imglist = os.listdir(root)
            for p in imglist:
                self.dataset.append((os.path.join(root,p),label))

    def __getitem__(self, index):
        img,label = self.dataset[index]
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img,(256,256))

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other):
        self.dataset.extend(other.dataset)
        return self