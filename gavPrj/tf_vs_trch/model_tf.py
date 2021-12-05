import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import cv2 as cv

# showprediction time

datasetFileName = "../master_dataset.npz"

with np.load(datasetFileName, allow_pickle=True) as data:
    dataImages = data['images']
    dataLabels = data['labels']
    dataLabelNames = data['labelnames']

desiredShape = (200, 200, 3)

N = len(dataImages)
shape = (N, desiredShape[0], desiredShape[1], desiredShape[2])

y = np.empty(shape, dtype='uint8')

for i in range(N):
    y[i] = cv.resize(dataImages[i], [200,200], interpolation=cv.INTER_NEAREST)

dataImages.dtype, y.dtype, y.shape

dataImages = y

dataLabels = dataLabels.astype('uint8')

trainImages, testImages, trainLabels, testLabels = train_test_split(dataImages, dataLabels, test_size=0.3, random_state=42)

classNames = sorted(np.unique(dataLabelNames))

inputShape = trainImages[0].shape
outputShape = len(classNames)

trainImages = trainImages / 255.0
testImages = testImages / 255.0

maxIterations = 5
testAccList = []
thresholdAcc = 0.8
lastTestAcc = 0.0

model = None
testLoss = 0.0
testAcc = 0.0
modelDir = 'tf'
epoch = 30

total_tr_start_time = time.time()
for iter in range(maxIterations):
    total_sim_start_time = time.time()
    print(f'simulation {iter + 1}', end='... ')

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=inputShape),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(outputShape)
    ])
    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # model training
    model.fit(trainImages, trainLabels, epochs=epoch, verbose=0)

    # model testing
    testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=0)

    # store the accuracy
    testAccList.append(testAcc)

    # print('\nTest accuracy:', testAcc)
    print(f'test accuracy {testAcc}', end='... ')
    total_sim_time_taken = time.time()-total_sim_start_time
    print(f'Total Simulation Time = {total_sim_time_taken}s')
    exportPath = ""

    # save model if greater than threshold-accuracy 0.95
    if testAcc > thresholdAcc:
        # SavedModel format
        version = f"cv_image_tf_512_{(testAcc*100):.0f}"

        # for SavedModel format
        exportPath = os.path.join(modelDir, version)
        # save the model
        model.save(exportPath, save_format="tf")
        # print(f'\nexport path = {exportPath}')
        print(f'export path = {exportPath}', end='')

        # # HDF5 format
        # exportPath = os.path.join(modelDir, f"{version}.h5")
        # # Save the entire model to a HDF5 file.
        # # The '.h5' extension indicates that the model should be saved to HDF5.
        # model.save(exportPath)
        # print("saved: ", exportPath)

        thresholdAcc = testAcc
        
    print('.')
total_tr_time_taken = time.time()-total_tr_start_time
print(f'Total Simulation Time: {total_tr_time_taken}s')   
