from os import listdir
from PIL import Image
import numpy as np
import random
# from skimage import transform,data
# img = data.camera()
# dst=transform.resize(img, (80, 60))
def reNamePostive():
    trainingFileList = listdir('positive')  #load the training set
    # print(trainingFileList)
    # i=0
    train = listdir('train')
    i=len(train)
    for img in trainingFileList:
        image=Image.open('positive/'+img)
        # image.resize((28, 28))
        print(np.array(image).shape)
        image.save('train/'+"1_"+str(i)+".BMP")
        # image2 = Image.open('postiveTestPhoto/'+"1_"+str(i))
        i+=1
        # print(array(image2).shape)
def reNameNegative():
    trainingFileList = listdir('negative')  #load the training set
    # print(trainingFileList)
    train = listdir('train')
    i = len(train)
    for img in trainingFileList:
        image=Image.open('negative/'+img)
        # image.thumbnail((28, 28),Image.ANTIALIAS)
        image.save('train/'+"0_"+str(i)+".BMP")
        # image2 = Image.open('postiveTestPhoto/'+"1_"+str(i))
        i+=1
def reSizeTrain(dir):
    trainingFileList = listdir(dir)  # load the training set
    # print(trainingFileList)
    # train = listdir('train')
    # i = len(train)
    i=0
    for img in trainingFileList:
        # print(img)
        image = Image.open(dir+'/' + img)
        image.thumbnail((32, 24),Image.ANTIALIAS)
        image.save(dir+"/"+img)
        # image2 = Image.open('postiveTestPhoto/'+"1_"+str(i))
        i += 1
def loadData(number):
    trainingFileList = listdir("E:/xinlun/train")
    n = len(trainingFileList)
    labels=np.zeros((number))
    train = np.zeros((number, 32,24,3))
    loadrange=random.sample(trainingFileList, number)
    # print(loadrange)
    j=0
    for i in loadrange:
        fr=Image.open("E:/xinlun/train/"+i)
        # fr1=fr.convert('1')
        # print(np.array(fr).shape)
        # if np.array(fr1).shape==(28,28):
        train[j, :] = np.array(fr).reshape([32,24,3])
        fileStr = i.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # if classNumStr==0:
        #    labels[j,0]=1
        # else:
        #     labels[j,1]=1
        labels[j]=classNumStr
        j+=1
    # labels=np.array(labels)
    # print(labels.shape)
    # labels=labels.reshape(128,1)
    return train,labels

def loadTestData(number):
    trainingFileList = listdir("E:/xinlun/test/")
    n = len(trainingFileList)
    labels=[]
    train = np.zeros((number, 32,24,3))
    loadrange=random.sample(trainingFileList, number)
    # print(loadrange)
    j=0
    for i in loadrange:
        fr=Image.open("E:/xinlun/test/"+i)
        # fr1=fr.convert('1')
        # print(np.array(fr).shape)
        # if np.array(fr1).shape==(28,28):
        train[j, :] = np.array(fr).reshape([32,24,3])
        fileStr = i.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        labels.append(classNumStr)
        j+=1
    labels=np.array(labels)
    # print(labels.shape)
    return train,labels

def reName(dir):
    trainingFileList = listdir(dir)  # load the training set
    # print(trainingFileList)
    # train = listdir('train')
    # i = len(train)
    i=0
    for img in trainingFileList:
        image = Image.open(dir+'/' + img)
        # image.thumbnail((28, 28),Image.ANTIALIAS)
        image.save(dir+'/' + "1_" + str(i) + ".BMP")
        # image2 = Image.open('postiveTestPhoto/'+"1_"+str(i))
        i += 1

def corp():
    trainingFileList = listdir('photo')
    print(trainingFileList)
    for img in trainingFileList:
        im = Image.open('photo/'+ img)
        # image.resize((28, 28))
        # image.save('testPhoto/' + "1_" + str(i), format="BMP")
        # 从左上角开始 剪切 200*200的图片
        img_size = im.size
        w = img_size[0] / 2.0
        h = img_size[1] / 2.0
        x = 0
        y = 0
        region = im.crop((x, y, x + w, y + h))
        region.save("corp/"+img+"crop-1.bmp")

        # 第2块
        x = w
        y = h
        region = im.crop((x, y, x + w, y + h))
        region.save("corp/"+img+"crop-2.bmp")

        # 第3块
        x = 0
        y = h
        region = im.crop((x, y, x + w, y + h))
        region.save("corp/"+img+"crop-3.bmp")

        # 第4块
        x = w
        y = 0
        region = im.crop((x, y, x + w, y + h))
        # region.save("corp/"+img+"crop-4.bmp")
# reName('positivetest')
# reSizeTrain("train")
# def loadTrainData():
#     train=listdir("positive/")
# corp()
# reSizePostive()
# reSizeNegative()
# train,labels=loadData(10)
# print(train,labels)

# loadData(10)