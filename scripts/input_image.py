# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:40:39 2019

@author: MP_lab_GPU
"""
#%%
from __future__ import print_function, division
from this import d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image


#repo_dir = 'C:\\Users\MP_lab_GPU\Desktop\Senior Design 2019\Senior Design\'

# CHANGE THIS DIRECTORY TO THE ML BREAST CANCER TOTAL FILES FOLDER
# repo_dir = r'C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\\'
reop_dir = r'C:\Users\Kris\..vs code files\senior design\ML-Breat_Cancer_Classfier-master'



#%%
imsize = 256
loader = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

# model = models.resnet152(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, 2)


#CHANGE PATH TO MODEL PATH, CHANGE MAP LOCATION BASED ON GPU
model.load_state_dict(torch.load(r'C:\Users\Kris\..vs code files\senior design\ML-Breat_Cancer_Classfier-master\model\\4-20_vgg_model_state_dict2.pt',
                                 map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), strict=False)
model.eval()
#CHANGE PATH LOCATION TO FOLDER WITH CROPPED MALIGNANT IMAGES
malignant_path = r'C:\Users\Kris\..vs code files\senior design\ML-Breat_Cancer_Classfier-master\images\Photos for Testing\data_test\Cancer'
#CHANGE PATH LOCATION TO FOLDER WITH CROPPED NONMALIGNANT IMAGES
non_malignant_path = r'C:\Users\Kris\..vs code files\senior design\ML-Breat_Cancer_Classfier-master\images\Photos for Testing\data_test\Not_Cancer'

#%%
#### THE FIRST ARGUMENT IS BENIGN, SECOND IS MALIGNANT
tp = [] #num correctly diagnosed malignant
fp = [] #num incorrectly diagnosed malignant
tn = [] #num correctly diagnosed benign
fn = [] #num incorrectly diagnosed negative
errors = 0
num_malignant = 0
num_non_malignant = 0

for i in os.listdir(malignant_path):
    try:
        image = image_loader(loader, os.path.join(malignant_path, i))
        
        y = model(image)
        if y.argmax().item() == 0:
            fn.append(i)
        else:
            tp.append(i)
        num_malignant +=1
    except:
        print(i)
        errors +=1
        
for i in os.listdir(non_malignant_path):
    try:
        image = image_loader(loader, os.path.join(non_malignant_path, i))
        num_non_malignant +=1
        y = model(image)
        if y.argmax().item() == 0:
            tn.append(i)
        else:
            fp.append(i)
    except:
        errors +=1

print("True Positives: " + str(len(tp)))
print("False Negatives: "+ str(len(fn)))
print("True Negatives: "+ str(len(tn)))
print("False Positives: "+ str(len(fp)))
print("Total Number of images: "+ str((len(tp)+len(tn)+len(fp)+len(fn))))
            
    
    
#%%







