import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#srcPaths = ('dataset/Screenshot1','dataset/Screenshot2','dataset/Screenshot3', 'dataset/Screenshot4')
#srcPaths = ('all_dataset/s1',
            # 'all_dataset/s10',
            # 'all_dataset/s11',
            # 'all_dataset/s12',
            # 'all_dataset/s13',
            # 'all_dataset/s14',
            # 'all_dataset/s15',
            # 'all_dataset/s16',
            # 'all_dataset/s17',
            # 'all_dataset/s18',
            # 'all_dataset/s19',
            # 'all_dataset/s2',
            # 'all_dataset/s20',
            # 'all_dataset/s21',
            # 'all_dataset/s22',
            # 'all_dataset/s23',
            # 'all_dataset/s24',
            # 'all_dataset/s25',
            # 'all_dataset/s26',
            # 'all_dataset/s27',
            # 'all_dataset/s28',
            # 'all_dataset/s29',
            # 'all_dataset/s3',
            # 'all_dataset/s30',
            # 'all_dataset/s31',
            # 'all_dataset/s32',
            # 'all_dataset/s33',
            # 'all_dataset/s34',
            # 'all_dataset/s35',
            # 'all_dataset/s36',
            # 'all_dataset/s37',
            # 'all_dataset/s38',
            # 'all_dataset/s39',
            # 'all_dataset/s4',
            # 'all_dataset/s40',
            # 'all_dataset/s41',
            # 'all_dataset/s42',
            # 'all_dataset/s43',
            # 'all_dataset/s44',
            # 'all_dataset/s45',
            # 'all_dataset/s46',
            # 'all_dataset/s47',
            # 'all_dataset/s48',
            # 'all_dataset/s49',
            # 'all_dataset/s5',
            # 'all_dataset/s50',
            # 'all_dataset/s51',
            # 'all_dataset/s52',
            # 'all_dataset/s53',
            # 'all_dataset/s54',
            # 'all_dataset/s55',
            # 'all_dataset/s56',
            # 'all_dataset/s57',
            # 'all_dataset/s58',
            # 'all_dataset/s59',
            # 'all_dataset/s6',
            # 'all_dataset/s60',
            # 'all_dataset/s61',
            # 'all_dataset/s62',
            # 'all_dataset/s63',
            # 'all_dataset/s7',
            # 'all_dataset/s8',
            # 'all_dataset/s9')
srcPaths = ('testdataset/t1','testdataset/t2')

datasetfilename = 'testdataset1.npz'


def create_dataset(datasetfilename, srcPaths, classNames):

    imgList = []
    labelList = []
    labelNameList = []

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
            labelList.append(classNames[label])
            labelNameList.append(label)

        # convert to imgList to numpy

        images = np.array(imgList, dtype='object')
        labels = np.array(labelList, dtype='object')
        labelnames = np.array(labelNameList)

        # save converted images and labels into compressed numpy zip file
    np.savez_compressed(datasetfilename, images=images, labels=labels, labelnames=labelnames)

    return True


def displayImg():

    # for fname in os.listdir(srcPath):

    pass


if __name__ == '__main__':
    # save a dataset in numpy compressed format

    # datasetfilename = 'tiredataset.npz'
    classNames = {'afiq':0, 'azureen':1, 'gavin':2, 'goke':3,  'inamul':4, 'jincheng':5, 'mahmuda':6, 'numan':7, 'saseendran':8}

    if create_dataset(datasetfilename, srcPaths, classNames):

        data = np.load(datasetfilename, allow_pickle=True)
        imgList = data['images']
        labelList = data['labels']
        labelNameList = data['labelnames']

        img = imgList[0]
        label = labelList[0]
        labelNameList = data['labelnames']

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
