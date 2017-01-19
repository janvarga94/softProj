import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from scipy import ndimage
import cv2
import random
import pytesseract
from keras.models import load_model
from keras.models import Sequential
from skimage.color import rgb2gray
import os
import sys
import math

model = load_model('trainedModel100-70-30-10')

def find_line(img):
    img = np.logical_and(img[:, :, 1] < 10, img[:, :,0] >100)
    firstIndex = None
    secondIndex = None
    for indx,row in enumerate(img):
        for indy,y in enumerate(row):
            if y:
                if firstIndex is None:
                    firstIndex = (indy,indx)
                secondIndex = (indy,indx)
    return firstIndex, secondIndex

def find_numbers(img):

    imageJustNum = np.logical_and(image[:, :, 0] > 0, image[:, :, 2] > 0)
    imageGray = rgb2gray(image)
    erozed = erozija(imageGray)

    labeled_img = label(erozed)
    regions = regionprops(labeled_img)

    brojevi = []

    for slika in regions:
        broj = []
        visina_sr = round((slika.bbox[0] + slika.bbox[2]) / 2)
        sirina_sr = round((slika.bbox[1] + slika.bbox[3]) / 2)
        DL1 = visina_sr - 14
        TR1 = visina_sr + 14
        DL2 = sirina_sr - 14
        TR2 = sirina_sr + 14

        broj.append(int(DL1))
        broj.append(int(DL2))
        broj.append(int(TR1))
        broj.append(int(TR2))

        broj.append(int(slika.bbox[0]))
        broj.append(int(slika.bbox[1]))
        broj.append(int(slika.bbox[2]))
        broj.append(int(slika.bbox[3]))

        brojevi.append(broj)


    print("brojevi len: {0}".format(len(brojevi)))

    for broj in brojevi:
        img_crop = imageGray[broj[0]: broj[2], broj[1]: broj[3]]
        t = model.predict(img_crop.reshape(1, 784), verbose=1)
        rez = np.argmax(t)
        if t[0,rez] > 0:
            print("na slici je: {0}".format(rez))
        plt.imshow(img_crop, 'gray')
        plt.show()


    return 0





def startDeleteion(img):
    rows = img.shape[0]
    cols = img.shape[1]

    matix_size = 5

    for row in range(0, rows - matix_size):
        for col in range(0, cols - matix_size):
            croped = img[row: row + matix_size, col: col + matix_size]
            sum = croped[:,matix_size - 1].sum() + croped[:,matix_size - 1].sum() + croped[matix_size - 1,:].sum() + croped[matix_size - 1,:].sum()
            if(sum < 0.1):
                img[row: row + matix_size - 1, col: col + matix_size - 1] = 0
            pass


def listContains(list, position):
    for (row,col) in list:
        if math.sqrt( (row - position[0])**2 + (col - position[1])**2 ) < 28:
            return True
    return False

def indexToRowAndCol(img, index):
    return index / (img.shape[1] - 28), index % (img.shape[1] - 28)

def processImage(img):
    # line = find_line(image)
    numbers = find_numbers(image)
    #plt.imshow(img)
    #plt.show()


def erozija(img):
    imageGray = rgb2gray(image)
    newOne = np.zeros((img.shape[0], img.shape[1]))
    rows = img.shape[0]
    cols = img.shape[1]
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if imageGray[row-1 : row + 1, col-1:col+1].mean() < 0.6:
                newOne[row,col] = 0
            else:
                newOne[row,col] = 1
    return newOne


if __name__ == "__main__":
   # plt.show(block=False)
    path = "videa\\video-0.avi"
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % 100 == 0:
            processImage(image)
            print(count)
        count += 1
    print("finished")


__name__ = "test"