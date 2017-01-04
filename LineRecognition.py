import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from scipy import ndimage
import cv2
import random
import pytesseract

def find_line(img):
    img = np.logical_and(img[:, :, 1] < 10, img[:, :,0] >100)
    firstIndex = None
    secondIndex = None
    for indx,row in enumerate(img):
        for indy,y in enumerate(row):
            if y == True:
                if(firstIndex == None):
                    firstIndex = (indy,indx)
                secondIndex = (indy,indx)
    return (firstIndex,secondIndex)

def find_numbers(img):
    pass

def processImage(img):
    plt.imshow(img)
    plt.draw()
    plt.show()
    line = find_line(image)
    numbers = find_numbers(image)

if __name__ == "__main__":
    plt.show(block=False)
    path = "videa\\video-0.avi"
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if(count % 100 == 0):
            processImage(image)
            print(count)
        count += 1
    print("finished")
