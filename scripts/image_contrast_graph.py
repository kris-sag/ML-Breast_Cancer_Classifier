#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:26:16 2020

@author: bencohen
"""
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
import os

# allow for device input, comment out to use shortcut
#directory = input("Enter directory path: ")
#os.chdir(directory)
#name = input("Enter image name: ")

#shortcut - change "directory", change file "name"
directory = r"C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\images\contrast_images"
os.chdir(directory)
name = 'clear'

if(os.path.exists(name + '.jpeg')):
    image = skimage.color.rgb2gray(skimage.io.imread(directory + "\\" + name + '.jpeg'))
    y = image[400,:]
    x = np.arange(len(y))

    plt.plot(x, y)
    plt.xlabel('Pixel')
    plt.ylabel('Greyscale Value')
    plt.title('Clear')
    plt.ylim([0, 1])
    plt.show()
else:
    print("File does not exist.")