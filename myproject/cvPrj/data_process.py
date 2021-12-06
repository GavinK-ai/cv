import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import cv2 as cv
import copy

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
    Labels = {cracked:0, normal:1}
    
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

def normalization(train_data,test_data):

    train_data = train_data/np.max(train_data)
    test_data = test_data/np.max(test_data)

    return train_data,test_data
    

def add_noise(train_data, train_label):

    noise_traindata = copy.deepcopy(train_data)
    noise_traindata = (noise_traindata + np.random.rand(*noise_traindata.shape)/1.1)
    noise_traindata = noise_traindata/np.max(noise_traindata)
    train_data = np.concatenate((train_data, noise_traindata))
    train_label = np.concatenate((train_label, train_label))
    
    return train_data, train_label

def greyscale(train_data, test_data):

    train_data = cv.cvtColor(train_data, cv.COLOR_BGR2GRAY)
    test_data = cv.cvtColor(test_data, cv.COLOR_BGR2GRAY)
    
    return train_data, test_data

def threshold():

    pass

def rotate(train_data, train_label):

    rot_traindata = copy.deepcopy(train_data)
    rot_traindata = cv.rotate(rot_traindata,  cv.ROTATE_90_CLOCKWISE)
    train_data = np.concatenate((train_data, rot_traindata))
    train_label = np.concatenate((train_label, train_label))

    return train_data, train_label

def flip(train_data, train_label):

    flip_traindata = copy.deepcopy(train_data)
    flip_traindata = cv.flip(flip_traindata, -1)
    train_data = np.concatenate((train_data, flip_traindata))
    train_label = np.concatenate((train_label, train_label))
    
    return train_data, train_label