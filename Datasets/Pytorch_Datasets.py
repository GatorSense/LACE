# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import PIL
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import datasets
import pdb

    
class FashionMNIST_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True): 
        
        self.transform = transform
        self.images = datasets.FashionMNIST(directory,train=train,transform=transform,
                                       download=download)

        self.targets = self.images.targets
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
class SVHN_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True):  
        
        self.transform = transform
        if train:
            self.split = 'train'
        else:
            self.split = 'test'
        self.images = datasets.SVHN(directory,split=self.split,transform=transform,
                                       download=download)

        self.targets = self.images.labels
        
        self.classes = np.unique(self.targets)
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
class CIFAR10_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True):  
        
        self.transform = transform
        self.images = datasets.CIFAR10(directory,train=train,transform=transform,
                                       download=download)

        self.targets = self.images.targets
        
        self.classes = self.images.classes
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
class CIFAR100_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True,coarse=False):  
        
        self.transform = transform
        if coarse:
            self.images = CIFAR100Coarse(directory,train=train,transform=transform,
                                           download=download)
        else:
            self.images = datasets.CIFAR100(directory,train=train,transform=transform,
                                           download=download)

        self.targets = self.images.targets
        
        self.classes = self.images.classes
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
class CIFAR100Coarse(CIFAR100):
    #Code from: https://github.com/ryanchankh/cifar100coarse
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13],dtype=np.float16)
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers',
                        'fruit and vegetables', 'household electrical devices',
                        'household furniture', 'insects', 'large carnivores',
                        'large man-made outdoor things', 'large natural outdoor scenes',
                        'large omnivores and herbivores', 'medium-sized mammals',
                        'non-insect invertebrates', 'people', 'reptiles',
                        'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

    
