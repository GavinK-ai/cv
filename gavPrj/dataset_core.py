import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

srcPaths = ('dataset/Screenshot1','dataset/Screenshot2')

datasetfilename = 'gavdataset.npz'

def create_dataset(datasetfilename, srcPaths):

    imgList = []
    labelList = []

    for srcPath in srcPaths:

        # append all files in srcPath dir into imgList and labelList

        for fname in os.listdir(srcPath):

            filePath = os.path.join(srcPath, fname)

            img = cv.imread(filePath)

            # spilt the last text in file name to save as label

            fname_no_ext = os.path.splitext(fname)[0]
            # label = fname_no_ext[-1]
            label = fname_no_ext

            imgList.append(img)
            labelList.append(label)
            
         

        # convert to imgList to numpy

        images = np.array(imgList, dtype='object')
        labels = np.array(labelList, dtype='object')

        # save converted images and labels into compressed numpy zip file
        np.savez_compressed(datasetfilename, images=images, labels=labels)

        return True

def displayImg():

    # for fname in os.listdir(srcPath):

    pass


if __name__ == '__main__':
    # save a dataset in numpy compressed format

    # datasetfilename = 'tiredataset.npz'

    if create_dataset(datasetfilename,srcPaths):

        data = np.load(datasetfilename, allow_pickle=True)
        imgList= data['images']
        labelList= data['labels'] 

        img = imgList[0]
        label = labelList[0]

        imgRGB = img[:, :, ::-1]

        plt.imshow(imgRGB)
        plt.title(label)

        plt.show()
    
    print(imgList.shape)
    print(labelList.shape)  

    # imgList, labelList = create_dataset()

    # img = imgList[0]
    # label = labelList[0]

    # imgRGB = img[:, :, ::-1]

    # plt.imshow(imgRGB)
    # plt.title(label)

    # plt.show()

    # img = imgList[1]
    # label = labelList[1]

    # imgRGB = img[:, :, ::-1]

    # plt.imshow(imgRGB)
    # plt.title(label)

    # plt.show()

    # img = imgList[3]
    # label = labelList[3]

    # imgRGB = img[:, :, ::-1]

    # plt.imshow(imgRGB)
    # plt.title(label)

    # plt.show()
