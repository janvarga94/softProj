import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import erosion,disk,opening
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
from VideoProcessor import *
from Models import *

model = load_model('trainedModel100-70-30-10')


def add(prev,next):
    return prev + next

def sub(prev, next):
    return prev - next

videoProcessorRedLine = VideoProcessor(add)
videoProcessorGreenLine = VideoProcessor(sub)

def find_line(img, bojaLinijeEnum = BojaLinije.CRVENA):
    """
    :return: pozicija linije, tako da prva pozicija je tacka levo, a druga tacka desno
    """
    if bojaLinijeEnum == BojaLinije.CRVENA:
        img = np.logical_and(img[:, :, 1] < 10, img[:, :,0] >100)
    elif bojaLinijeEnum == BojaLinije.ZELENA:
        img = np.logical_and(img[:, :, 0] < 10, img[:, :, 1] > 100)

    tacka1 = None
    tacka2 = None
    for indrow,row in enumerate(img):
        for indcol,y in enumerate(row):
            if y:
                if tacka1 is None:
                    tacka1 = Tacka(indrow,indcol)
                tacka2 = Tacka(indrow,indcol)
    if(tacka1.col < tacka2.col):
        return Pozicija(tacka1.row, tacka1.col, tacka2.row, tacka2.col)
    else:
        return Pozicija(tacka2.row, tacka2.col, tacka1.row, tacka1.col)

def find_numbers(img):

    imageJustNum = np.logical_and(image[:, :, 0] > 0, image[:, :, 2] > 0)
    imageGray = rgb2gray(image)
    erozed = brisanjeZvezdica(imageGray)

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

        brojevi.append(broj)

    rezultat = [[] for x in range(10)]

    for broj in brojevi:
        img_crop = erozed[broj[0]: broj[2], broj[1]: broj[3]]
        reshaped = None
        try:
            reshaped = img_crop.reshape(1, 784)
        except Exception:
            continue
        t = model.predict(reshaped, verbose=0)
        rez = np.argmax(t)
        if t[0,rez] > 0:
            rezultat[rez].append(Pozicija(broj[0],broj[1],broj[2],broj[3]))
            plt.plot([broj[1]],[broj[0]], 'ys')

    return rezultat

def processImage(img):
    linijaCrvena = find_line(img, BojaLinije.CRVENA)
    linijaZelena = find_line(img, BojaLinije.ZELENA)
    numbers = find_numbers(img)

    if not videoProcessorRedLine.IsLinijaSet():
        videoProcessorRedLine.SetLiniju(linijaCrvena)

    if not videoProcessorGreenLine.IsLinijaSet():
        videoProcessorGreenLine.SetLiniju(linijaZelena)

    videoProcessorRedLine.Process(numbers)
    videoProcessorGreenLine.Process(numbers)

   # plt.imshow(img,'gray');
   # plt.show()

#nekoristim
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

def brisanjeZvezdica(img):
    imageGray = rgb2gray(image)
    mask = imageGray[:,:] > 0.5
    zeroOne = np.zeros((img.shape[0], img.shape[1]))
    zeroOne[mask] = 1
  #  selem = disk(1)
  #  eroded = opening(zeroOne, selem)
    labeled_img = label(zeroOne)
    regions = regionprops(labeled_img)

    for slika in regions:
        broj = []
        visina_sr = round(slika.bbox[2] - slika.bbox[0])
        sirina_sr = round(slika.bbox[3] - slika.bbox[1])
        if visina_sr < 10 and sirina_sr < 10:
            zeroOne[slika.bbox[0]:slika.bbox[2],slika.bbox[1]:slika.bbox[3]] = 0

    return zeroOne


if __name__ == "__main__":
   # plt.show(block=False)
    path = "videa\\2linije\\video-3.avi"
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % 30 == 0:
            processImage(image)

        count += 1
    print("done")
    print("suma kroz crvenu = {0}".format(videoProcessorRedLine.trenutnaVrednostAgregacije))
    print("-suma kroz zelenu = {0}".format(videoProcessorGreenLine.trenutnaVrednostAgregacije))


