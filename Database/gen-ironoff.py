import sys, getopt

import argparse

import sys
import os
import time

import numpy as np

import pandas as pd

from PIL import Image, ImageChops
import PIL.ImageOps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot  as pyplot
import gzip

import theano
import theano.tensor as T

import lasagne
import nolearn

import pickle, gzip


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def scale(image, max_size, method=Image.ANTIALIAS):
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)), method)
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]), method)

    offset = ( int((max_size[0] - scaled.size[0]) / 2), int((max_size[1] - scaled.size[1]) / 2) )
    back = Image.new('L', max_size)
    back.paste(scaled, offset)
    return back

pathIRONOFF = './BaseIronoff'

filesIRONOFF_Dig_train = pathIRONOFF + '/Dbd/Digit/Digit_train_tif.dbd'
dfIRONOFF_Dig_train = pd.read_csv(filesIRONOFF_Dig_train,sep=' ', names=['Path', 'Content'])
filesIRONOFF_Dig_test = pathIRONOFF + '/Dbd/Digit/Digit_test_tif.dbd'
dfIRONOFF_Dig_test = pd.read_csv(filesIRONOFF_Dig_test,sep=' ', names=['Path', 'Content'])

filesIRONOFF_Char_train = pathIRONOFF + '/Dbd/Character/Character_train_tif.dbd'
dfIRONOFF_Char_train = pd.read_csv(filesIRONOFF_Char_train,sep=' ', names=['Path', 'Content'])
filesIRONOFF_Char_test = pathIRONOFF + '/Dbd/Character/Character_test_tif.dbd'
dfIRONOFF_Char_test = pd.read_csv(filesIRONOFF_Char_test,sep=' ', names=['Path', 'Content'])

filesIRONOFF_FrWord_train = pathIRONOFF + '/Dbd/FrenchWord/FrenchWord_train_tif.dbd'
dfIRONOFF_FrWord_train = pd.read_csv(filesIRONOFF_FrWord_train,sep=' ', names=['Path', 'Content'], encoding='latin')
filesIRONOFF_FrWord_test = pathIRONOFF + '/Dbd/FrenchWord/FrenchWord_test_tif.dbd'
dfIRONOFF_FrWord_test = pd.read_csv(filesIRONOFF_FrWord_test,sep=' ', names=['Path', 'Content'], encoding='latin')

filesIRONOFF_EnWord_train = pathIRONOFF + '/Dbd/EnglishWord/EnglishWord_train_tif.dbd'
dfIRONOFF_EnWord_train = pd.read_csv(filesIRONOFF_EnWord_train,sep=' ', names=['Path', 'Content'], encoding='utf-8')
filesIRONOFF_EnWord_test = pathIRONOFF + '/Dbd/EnglishWord/EnglishWord_test_tif.dbd'
dfIRONOFF_EnWord_test = pd.read_csv(filesIRONOFF_EnWord_test,sep=' ', names=['Path', 'Content'], encoding='utf-8')

dfIRONOFF_train = dfIRONOFF_Char_train
dfIRONOFF_test = dfIRONOFF_Char_test

dbs = [dfIRONOFF_Dig_train, dfIRONOFF_Dig_test, dfIRONOFF_Char_train, dfIRONOFF_Char_test, dfIRONOFF_FrWord_train, dfIRONOFF_FrWord_test, dfIRONOFF_EnWord_train, dfIRONOFF_EnWord_test]

Type = ["Dig_train", "Dig_test", "Char_train", "Char_test", "FrWord_train", "FrWord_test", "EnWord_train", "EnWord_test"]
Names = ['Path', 'Content', 'PathAbs','Type', 'SizeX', 'SizeY']
DB = pd.DataFrame()

# for d in range(len(dbs)):
#     db = dbs[d]
#     typeImg = np.repeat(Type[d], len(db))
#     db['Type'] = typeImg
#     
#     truePath = []
#     #imgSize = list()
#     imgSizeX = []
#     imgSizeY = []
#     for i in range(len(db)):
#         path = db.get_value(i, 'Path')
#         content = db.get_value(i, 'Content')
#         f=os.path.basename(path.replace("\\", "/"))
#         c=f[0]
#         d=os.path.dirname(path.replace("\\", "/")).replace("PATH", pathIRONOFF+"/data/"+ c)
#         file = d + "/" + f
#         truePath = np.concatenate((truePath, [file]))
#         #os.system("identify file")
#         img = Image.open(str(file)).convert('L')
#      #   imgSize = np.concatenate((imgSize, img.size), axis=1)
#         imgSizeX = np.concatenate((imgSizeX, [img.size[0]]))
#         imgSizeY = np.concatenate((imgSizeY, [img.size[1]]))
#     #db['Size'] = imgSize
#     db['SizeX'] = imgSizeX
#     db['SizeY'] = imgSizeY
#     db['PathAbs'] = truePath
#     DB = DB.append(db)
# DB = DB.reset_index()
# DB = DB.drop('index', 1)
# DB.to_csv('./dbIRONOFF.csv', sep=";")


dbcsv = pd.read_csv('./dbIRONOFF.csv',sep=';', names=['Path', 'Content'], encoding='utf-8')

classes = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9","euro"]
intclasses = np.arange(len(classes))

XTrain = np.array([]).reshape((0, 1, 256, 256))
yTrain = []
cTrain = []
df = dfIRONOFF_train
pklname= 'ironoff_character.pkl.gz'
for i in np.arange(len(df)):
    path = df.get_value(i, 'Path')
    content = df.get_value(i, 'Content')
    classi = classes.index(content)
    #print(path, content)
    f=os.path.basename(path.replace("\\", "/"))
    c=f[0]
    d=os.path.dirname(path.replace("\\", "/")).replace("PATH", pathIRONOFF+"/data/"+ c)
    file = d + "/" + f
    img = Image.open(str(file)).convert('L')
    imgInv = PIL.ImageOps.invert(img)
    imgScale = scale(imgInv, [256, 256] )
#     imgInv=imgInv.resize([167, 214])
#     imgCrop = trim (imgInv)
#     arr = np.array(imgInv)
    arr = np.array(imgScale).reshape( (1,1,256,256) )
    XTrain = np.concatenate((XTrain, arr), axis=0)
    yTrain.append(content)
    cTrain.append(classi)
    
XTest = np.array([]).reshape((0 , 1, 256, 256))
# XTest = []
yTest = []
cTest = []
df = dfIRONOFF_test
for i in np.arange(len(df)):
    path = df.get_value(i, 'Path')
    content = df.get_value(i, 'Content')
    classi = classes.index(content)
    #print(path, content)
    f=os.path.basename(path.replace("\\", "/"))
    c=f[0]
    d=os.path.dirname(path.replace("\\", "/")).replace("PATH", pathIRONOFF+"/data/"+ c)
    file = d + "/" + f
    img = Image.open(str(file)).convert('L')
    imgInv = PIL.ImageOps.invert(img)
    imgScale = scale( imgInv, [256, 256] )
#     imgInv = imgInv.resize([167, 214])
#     imgCrop = trim (imgInv)
    arr = np.array(imgScale).reshape( (1,1,256,256) )
    XTest = np.concatenate((XTest, arr), axis=0)
#     XTest.append(arr)
    yTest.append(content)
    cTest.append(classi)

print(np.array(XTrain).shape)
print(np.array(XTest).shape)
print(np.array(yTrain).shape)
print(np.array(yTest).shape)
listData = [XTrain, yTrain, cTrain, XTest, yTest, cTest]

with gzip.open(pklname, "wb") as f:
    pickle.dump(listData, f)

