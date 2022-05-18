import cv2
import os
import glob
import re
import numpy as np

import argparse
import skimage.io as io
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

print(tf.__version__)

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

import imageio.core.util

tesela=256; # image size to predict in pixels

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

np.seterr(divide='ignore', invalid='ignore')

def num_sort(test_string):
    return list(map(int, re.findall(r'(?<=split_)\d+', test_string)))[0]


# Data to predict

path=os.path.abspath(os.getcwd())
pathDataImg=os.path.join(path,'DataToClassify')
for filenameR in glob.glob("DataToClassify/.*"):
        os.remove(filenameR)

file_list=os.listdir(pathDataImg)
file_list=sorted(file_list)

result = np.zeros(shape=(len(file_list),1))
result = result.astype('str')

for i1 in range(0,len(file_list),1):

# Divide code
	
	# Create the results folder for predicted images
    if not os.path.exists('Results'):
        os.mkdir('Results')

    imgOriginal = cv2.imread('DataToClassify/%s' %file_list[i1])
    height, width, channels = imgOriginal.shape
    pathResult=os.path.join(path,'Results/%s' %file_list[i1])
    cv2.imwrite(pathResult, imgOriginal)

    originalSizeX=width; originalSizeY=height; num=1;
    xMax=np.ceil(originalSizeX/tesela)
    yMax=np.ceil(originalSizeY/tesela)
    numMaxImage=int(xMax*yMax)

    [imName,imExt] = os.path.splitext("%s" %file_list[i1])

    # Split the image in tesela size images
    if not os.path.exists('PredDivided'):
        os.mkdir('PredDivided')
    else:
        for filenameR in glob.glob("PredDivided/*"):
            os.remove(filenameR)
    pathPredDivided=os.path.join(path,'PredDivided')
    for i2 in range(0,originalSizeY,tesela):
        for j2 in range(0,originalSizeX,tesela):
            cropped_image = imgOriginal[i2:i2+tesela, j2:j2+tesela]
            # Save the cropped image
            pathPredDivided1=os.path.join(pathPredDivided,'%s_split_%d.jpg' %(imName,num))
            cv2.imwrite(pathPredDivided1, cropped_image)
            num=num+1

    print("\n")
    print('%s Divided' %imName)

    # VGG19 CNN DL algorithm

    sx=256; sy=256; ch=3;

    model = keras.models.load_model('RGB_RJBDA_VGG19_PreTrained_Ref_5C_1.h5')

    # Load prediction data

    pathDataPred=os.path.join(path,'PredDivided/')
    for filenameR in glob.glob("PredDivided/.*"):
        os.remove(filenameR)
    file_listPred=os.listdir(pathDataPred)
    #file_listPred=sorted(file_listPred)
    file_listPred.sort(key=num_sort) 

    samples = len(file_listPred)
    pred_images=np.zeros(shape=(samples,sy,sx,ch))
    num=0
    for iFile in range(0,len(file_listPred),1):
      image = Image.open('PredDivided/%s' %file_listPred[iFile])
      imageR = image.resize((sx,sy))
      image_array = np.array(imageR)
      image_array = image_array.reshape((sy,sx,ch))
      pred_images[num]=image_array
      num=num+1

    class_names = ['Avena','Hordeum','Triticum', 'Background', 'Artifact']

    pred_images = pred_images.astype('uint8')

    pred = model.predict(pred_images, batch_size=1)
    pred2 = np.zeros(shape=(1,len(pred)))
    pred3= pred2.astype(str)
    print("\n")
    print("%s Prediction:" %imName)
    num0=0; num1=0; num2=0; num3=0; num4=0; iPlotX = 0; iPlotY = 0;
    for iPred in range(0, len(pred), 1):
        pred2[0,iPred] = np.argmax(pred[iPred])
        [imNamePred,imExtPred] = os.path.splitext("%s" %file_listPred[iPred])
        pred3[0,iPred] = str(imNamePred) + ": "+ str(class_names[int(pred2[0,iPred])])
        print(pred3[0,iPred])
        
        imgResult = cv2.imread('Results/%s' %file_list[i1])

        if pred2[0,iPred] == 0:
            num0 = num0 + 1
            cv2.rectangle(imgResult, pt1=((tesela*iPlotX)+5, (tesela*iPlotY)+5), pt2=((tesela*(iPlotX+1))-5, (tesela*(iPlotY+1))-5), color=(0,0,255), thickness=10)

        elif pred2[0,iPred] == 1:
            num1 = num1 + 1
            cv2.rectangle(imgResult, pt1=((tesela*iPlotX)+5, (tesela*iPlotY)+5), pt2=((tesela*(iPlotX+1))-5, (tesela*(iPlotY+1))-5), color=(0,255,0), thickness=10)

        elif pred2[0,iPred] == 2:
            num2 = num2 + 1
            cv2.rectangle(imgResult, pt1=((tesela*iPlotX)+5, (tesela*iPlotY)+5), pt2=((tesela*(iPlotX+1))-5, (tesela*(iPlotY+1))-5), color=(255,0,0), thickness=10)

        elif pred2[0,iPred] == 3:
            num3 = num3 + 1
        else:
            num4 = num4 + 1

        cv2.imwrite(pathResult, imgResult)
        iPlotX = iPlotX + 1
        if iPlotX == 10:
            iPlotX = 0
            iPlotY = iPlotY + 1

    print("Avena: ", num0)
    print("Hordeum: ", num1)
    print("Triticum: ", num2)
    print("Background: ", num3)
    print("Artifact: ", num4)

    if num0 > num1 and num0 > num2:
        result[i1,0] = "The phyto %s is Avena" %imName
        print("The phyto %s is Avena" %imName)

    elif num1 > num0 and num1 > num2:
        result[i1,0] = "The phyto %s is Hordeum" %imName
        print("The phyto %s is Hordeum" %imName)

    elif num2 > num0 and num2 > num1:
        result[i1,0] = "The phyto %s is Triticum" %imName
        print("The phyto %s is Triticum" %imName)

    elif num1 != num0 and num0 == num2:
        result[i1,0] = "The phyto %s is A T" %imName
        print("The phyto %s has the same probability to be Avena and Triticum" %imName)

    elif num1 != num0 and num1 == num2:
        result[i1,0] = "The phyto %s is H T" %imName
        print("The phyto %s has the same probability to be Hordeum and Triticum" %imName)

    elif num1 == num0 and num1 != num2:
        result[i1,0] = "The phyto %s is A H" %imName
        print("The phyto %s has the same probability to be Avena and Hordeum" %imName)

    else:
        result[i1,0] = "The phyto %s is all" %imName
        print("The phyto %s has the same probability to be Avena, Hordeum and Triticum" %imName)

print("\n")
print("Classification:")
np.savetxt('Results.txt', result, fmt='%s', delimiter=',')
for i in range(0,len(result),1):
    print(result[i][0])




