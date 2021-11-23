import os
import cv2 as cv
import matplotlib.pyplot as plt


def create_dataset():

    imgList = []
    labelList = []

    # read image path from folder

    srcPath = 'rawdata'

    for fname in os.listdir(srcPath):

        filePath = os.path.join(srcPath, fname)

        img = cv.imread(filePath)

        fname_no_ext = os.path.splitext(fname)[0]
        label = fname_no_ext[-1]


        # label = fname[-1]

        imgList.append(img)
        labelList.append(label)

    return imgList, labelList


if __name__ == '__main__':

    imgList, labelList = create_dataset()

    img = imgList[0]
    label = labelList[0]

    imgRGB = img[:, :, ::-1]

    plt.imshow(imgRGB)
    plt.title(label)

    plt.show()

    img = imgList[1]
    label = labelList[1]

    imgRGB = img[:, :, ::-1]

    plt.imshow(imgRGB)
    plt.title(label)

    plt.show()

    img = imgList[3]
    label = labelList[3]

    imgRGB = img[:, :, ::-1]

    plt.imshow(imgRGB)
    plt.title(label)

    plt.show()
