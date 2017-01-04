#!/usr/bin/python

import numpy as np

from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import label, regionprops
from sklearn.datasets import fetch_mldata


def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

digits = fetch_mldata('MNIST original',data_home='mnist')

data   = digits.data / 255.0
labels = digits.target.astype('int')

train_rank = 10000
test_rank = 100

train_subset = np.random.choice(data.shape[0], train_rank)
test_subset = np.random.choice(data.shape[0], test_rank)


train_data = data
train_labels = labels

def padwithtens(vector, pad_width, iaxis, kwargs):
     vector[:pad_width[0]] = 0
     vector[-pad_width[1]:] = 0
     return vector

trening = []
print("obradjujem slike")
for index in range(0,len(train_data)):
    image = train_data[index].reshape(28,28)
    newImg = np.lib.pad(image, (30, 30), padwithtens)

    imageJustNum = newImg[:, :] > 0

    labeled_img = label(imageJustNum)
    regions = regionprops(labeled_img)

    visina_sr = round((regions[0].bbox[0] + regions[0].bbox[2]) / 2)
    sirina_sr = round((regions[0].bbox[1] + regions[0].bbox[3]) / 2)
    DL1 = visina_sr - 14
    TR1 = visina_sr + 14
    DL2 = sirina_sr - 14
    TR2 = sirina_sr + 14

    img = newImg[regions[0].bbox[0] : regions[0].bbox[2], regions[0].bbox[1] : regions[0].bbox[3]]

    img_crop = newImg[int(DL1): int(TR1), int(DL2): int(TR2)]
    # plt.imshow(img_crop, 'gray')
    # plt.show()

    trening.append(img_crop.reshape(784))
print('slike obradjene')
trening = np.array(trening)

print(trening.shape)
# test dataset
test_data = data[test_subset]
test_labels = labels[test_subset]

train_out = to_categorical(train_labels, 10)
test_out = to_categorical(test_labels, 10)

model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('softmax'))



sgd = SGD(lr=0.01, decay=1e-5, momentum=0, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.load_weights('trainedModel')
training = model.fit(trening, train_out, nb_epoch=1, batch_size=400, verbose=1)

try:
    model.save("trainedModel")
except:
    pass



print(training.history['loss'][-1])

# getting parts of image
"""
all_num=[]
for slika_ind in range(0, 100):
    image = imread('images/img-'+str(slika_ind)+'.png')
    imageJustNum = np.logical_and(image[:,:,0]>0, image[:,:,2]>0)
    imageGray = rgb2gray(image)

    combImage = np.multiply(imageGray,imageJustNum)

    # plt.imshow(imageJustNum,'gray')
    # plt.show()

    labeled_img = label(imageJustNum)
    regions = regionprops(labeled_img)

    slike = []
    for region in regions:
        bbox = region.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        if np.logical_or(h >= 3, w >= 3):
            slike.append(region)

    brojevi = []

    for slika in slike:
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

    suma = 0.0
    for broj in brojevi:
        img_crop = combImage[broj[0]: broj[2], broj[1]: broj[3]]
        t = model.predict(img_crop.reshape(1, 784), verbose=1)
        vrednost = np.argmax(t)
        # print 'Slika: '+str(slika_ind)+' Vrednost: ' + str(vrednost)
        # plt.imshow(img_crop, 'gray')
        # plt.show()


        suma = suma+vrednost

    all_num.append(suma)

f = open('out.txt', 'w')
f.writelines('RA 22-2013 Svetozar Stojkovic\nfile\tsum\n')
for curr in range(0, 100):
    f.writelines('images/img-' + str(curr) + '.png\t' + str(all_num[curr]) + '\n')

f.close()
"""
# import sys
#
# res = {}
# n = 0
# with open('res.txt') as file:
#     data = file.read()
#     lines = data.split('\n')
#     for id, line in enumerate(lines):
#         if(id>0):
#             cols = line.split('\t')
#             if(cols[0] == ''):
#                 continue
#             cols[1] = cols[1].replace('\r', '')
#             res[cols[0]] = cols
#             n += 1
#
# correct = 0
# student = []
# with open("out.txt") as file:
#     data = file.read()
#     lines = data.split('\n')
#     for id, line in enumerate(lines):
#         cols = line.split('\t')
#         if(cols[0] == ''):
#             continue
#         if(id==0):
#             student = cols
#         elif(id>1):
#             cols[1] = cols[1].replace('\r', '')
#             if (res[cols[0]] == cols):
#                 correct += 1
#
# print student
# print 'Tacnih:\t'+str(correct)
# print 'Ukupno:\t'+str(n)
# print 'Uspeh:\t'+str(100*correct/n)+'%'