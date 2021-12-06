import os
import time

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import copy

from data_process import create_data, normalization, add_noise, greyscale,rotate,flip

# Making function for create data from path

trainImages, trainLabels = create_data(test_data=False)
testImages, testLabels = create_data(test_data=True)

# convert to grayscale
# trainImages,testImages = greyscale(trainImages,testImages)

# Normalized Image
trainImages,testImages = normalization(trainImages, testImages)

# Add noise to duplicate data
#trainImages,trainLabels = add_noise(trainImages, trainLabels)

# Rotate image by 90 degrees clockwise
trainImages,trainLabels = rotate(trainImages, trainLabels)

# Flip image horizontally and vertically
trainImages,trainLabels = flip(trainImages, trainLabels)

input_shape = trainImages[0].shape

maxIterations = 5
testAccList = []
thresholdAcc = 0.65
lastTestAcc = 0.0

testLoss = 0.0
testAcc = 0.0
modelDir = 'tf'
epoch = 50

total_tr_start_time = time.time()
for iter in range(maxIterations):
    total_sim_start_time = time.time()
    print(f'Simulation {iter + 1}', end='... ')

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # model training
    model.fit(trainImages, trainLabels, epochs=epoch, verbose=1)

    # model testing
    testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=1)

    # store the accuracy
    testAccList.append(testAcc)

    # print('\nTest accuracy:', testAcc)
    print(f'Test Accuracy {testAcc*100:.3f}', end='... ')
    total_sim_time_taken = time.time()-total_sim_start_time
    print(f'Total Simulation Time = {total_sim_time_taken:0.3f}s')
    exportPath = ""

    # save model if greater than threshold-accuracy 0.95
    if testAcc > thresholdAcc:
        # SavedModel format
        version = f"cv_tire_tf_{(testAcc*100):.0f}"

        # for SavedModel format
        exportPath = os.path.join(modelDir, version)
        # save the model
        model.save(exportPath, save_format="tf")
        # print(f'\nexport path = {exportPath}')
        print(f'Export path = {exportPath}', end='')

        # # HDF5 format
        # exportPath = os.path.join(modelDir, f"{version}.h5")
        # # Save the entire model to a HDF5 file.
        # # The '.h5' extension indicates that the model should be saved to HDF5.
        # model.save(exportPath)
        # print("saved: ", exportPath)

        thresholdAcc = testAcc
        
total_tr_time_taken = time.time()-total_tr_start_time
print(f'Total Simulation Time: {total_tr_time_taken:.3f}s')   
