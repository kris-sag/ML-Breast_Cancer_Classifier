import os, glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

# allow for device input, comment out to use shortcut
#directory = input("Enter directory path: ")
#os.chdir(directory)
#shortcut - change directory to the path of folder holding images
os.chdir(r"C:\Users\joekh\Documents\GitHub\ML-Breat_Cancer_Classfier\images\Photos for Testing\CroppedImages Malignant\\")



jpgImages = glob.glob("*.jpg")
pngImages = glob.glob("*.png")
images = jpgImages + pngImages

if len(images) < 1:
    print('No images to crop, program ending...')
    exit()

cwd = os.getcwd()
identifyFolder = cwd[::-1]
folder = ''

for c in identifyFolder:
    if c == '\\':
        break
    folder += c

folder = folder[::-1]

print(folder)
os.chdir('..')

folder = 'CroppedImages ' + folder

if not(os.path.isdir(folder)):
    os.mkdir(folder)
    os.chdir(folder)
else:
    os.chdir(folder)

outputDir = os.getcwd()

for imageFile in images:
    os.chdir(cwd)
    print('Found: ' + imageFile)
    image = cv2.imread(imageFile)
    image_src = image.copy()
    flat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(flat, 40, 255, cv2.THRESH_BINARY_INV)
    mask = np.zeros_like(image)
    mask[:,:,0] = thresh
    mask[:,:,1] = thresh
    mask[:,:,2] = thresh

    outImage = image | mask
    os.chdir(outputDir)
    cv2.imwrite(imageFile, outImage)


