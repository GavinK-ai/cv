

import dataset_core
import numpy as np
import matplotlib.pyplot as plt

fileName = 'tiredataset.npz'

if __name__ == '__main__':

    # Load data from dataset
    data = np.load(fileName, allow_pickle=True)

    #Display image


    imgList = data['images']
    labelList = data['labels']

    img = imgList[0]
    label = labelList[0]

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

    img = imgList[2]
    label = labelList[2]

    imgRGB = img[:, :, ::-1]

    plt.imshow(imgRGB)
    plt.title(label)

    plt.show()
