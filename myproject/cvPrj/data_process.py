import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import cv2 as cv
import copy
from pathlib import Path
import pandas as pd
import glob

img_size = 200


def create_data(test_data=False):

    # Checking for test or train data conditions

    # Test data
    if test_data:
        cracked = 'dataset/testing_data/cracked'
        normal = 'dataset/testing_data/normal'

    # Train data
    else:
        cracked = 'dataset/training_data/cracked'
        normal = 'dataset/training_data/normal'

    # Labels for train and test
    Labels = {cracked: 0, normal: 1}

    # Initializing list for storing train and test data
    data = []

    # Initializing list for storing train and test label
    labels = np.array([])

    # Looping through each label
    for label in Labels:

        # Looping through cracked train data
        for ls in os.listdir(label):

            # Join each ls element with file path
            path = os.path.join(label, ls)

            # Read images from path using cv
            img = cv.imread(path, cv.IMREAD_COLOR)
            img = cv.resize(img, (img_size, img_size))

            # Adding data into data and labels list
            data.append(np.array(img))
            labels = np.append(labels, Labels[label])

    return np.array(data), labels


def normalization(train_data, test_data):

    train_data = train_data/np.max(train_data)
    test_data = test_data/np.max(test_data)

    return train_data, test_data


def add_noise(train_data, train_label):

    noise_traindata = copy.deepcopy(train_data)
    noise_traindata = (noise_traindata +
                       np.random.rand(*noise_traindata.shape)/1.1)
    noise_traindata = noise_traindata/np.max(noise_traindata)
    train_data = np.concatenate((train_data, noise_traindata))
    train_label = np.concatenate((train_label, train_label))

    return train_data, train_label


def greyscale(train_data, test_data):

    gray_train_data_list = []
    gray_test_data_list = []

    for i in range(len(train_data)):
        gry_traindata = copy.deepcopy(train_data)
        gray_train_image = cv.cvtColor(gry_traindata, cv.COLOR_BGR2GRAY)
        train_data = gray_train_data_list.append(gray_train_image)

    for i in range(len(test_data)):   
        gry_testdata = copy.deepcopy(test_data)
        gray_test_image = cv.cvtColor(gry_testdata, cv.COLOR_BGR2GRAY)
        test_data = gray_test_data_list.append(gray_test_image)

    return train_data, test_data


def threshold(train_data, test_data):

    thresh = 50
    maxval = 255

    thres_train_data_list = []
    thres_test_data_list = []

    for i in range(len(train_data)):
        thres_traindata = copy.deepcopy(train_data)
        ret, train_data_threshed = cv.threshold(
            thres_traindata, thresh, maxval, cv.THRESH_BINARY)
        train_data = gray_train_data_list.append(train_data_threshed)

    for i in range(len(test_data)):  
        thres_testdata = copy.deepcopy(test_data) 
        ret, test_data_threshed = cv.threshold(
            thres_testdata, thresh, maxval, cv.THRESH_BINARY)
        test_data = gray_test_data_list.append(test_data_threshed)

    return train_data, test_data

def rotate(train_data, train_label):

    rot_traindata_list = []
    for i in range(len(train_data)):
        rot_traindata = train_data[i].copy()
        # rot_traindata = cv.rotate(rot_traindata,cv.ROTATE_90_CLOCKWISE)
        rot_traindata = np.rot90(rot_traindata)
        rot_traindata_list.append(rot_traindata)

    train_data = np.concatenate((train_data, rot_traindata_list))
    train_label = np.concatenate((train_label, train_label))

    return train_data, train_label


def flip(train_data, train_label):

    flip_traindata_list = []
    for i in range(len(train_data)):
        flip_traindata = train_data[i].copy()
        # rot_traindata = cv.rotate(rot_traindata,cv.ROTATE_90_CLOCKWISE)
        flip_traindata = cv.flip(flip_traindata, -1)
        flip_traindata_list.append(flip_traindata)

    train_data = np.concatenate((train_data, flip_traindata_list))
    train_label = np.concatenate((train_label, train_label))

    return train_data, train_label

def process(data):
    # For resnet model data import
    path=Path(data)#converting the string to path
    filepaths=list(path.glob(r"*/*.jpg"))#Going through all the subpaths 
    labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))#Separating the label from filepath and storing it
    df1=pd.Series(filepaths,name='filepaths').astype(str)
    df2=pd.Series(labels,name='labels')
    df=pd.concat([df1,df2],axis=1)#Making the dataframe
    return df