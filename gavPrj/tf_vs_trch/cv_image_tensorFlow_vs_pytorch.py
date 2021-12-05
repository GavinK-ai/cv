import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2 as cv
from model_torch import NeuralNetwork, test
import time

datasetFileName = "../master_dataset.npz"

#tfPath = 'tf/cv_image_tf_97'
tfPath = 'tf/cv_image_tf_512_99'
tfModel = tf.keras.models.load_model(tfPath)


trchPath = 'trch/cv_image_torch_93.pth'
model = NeuralNetwork()
model.load_state_dict(torch.load(trchPath))

# print(tfModel.summary())
# print(trchModel)


def importImageData(datasetFileName):

    with np.load(datasetFileName, allow_pickle=True) as data:
        dataImages = data['images']
        dataLabels = data['labels']
        dataLabelNames = data['labelnames']

    desiredShape = (200, 200, 3)

    N = len(dataImages)
    shape = (N, desiredShape[0], desiredShape[1], desiredShape[2])

    y = np.empty(shape, dtype='uint8')

    for i in range(N):
        y[i] = cv.resize(dataImages[i], [200, 200],
                         interpolation=cv.INTER_NEAREST)

    dataImages = y
    dataLabels = dataLabels.astype('uint8')

    return dataImages, dataLabels


def tensorflowPredict(dataImage, dataLabels, i):

    testImage = dataImage / 255.0
    testLabel = dataLabels

    tf_start_time = time.time()

    predictions = tfModel.predict(testImage)
    predictedLabel = np.argmax(predictions[i])
    tf_infer_time = time.time()-tf_start_time
    print(predictedLabel, testLabel[i], predictedLabel == testLabel[i])
    print(
        f'Predicted Class: {classes[predictedLabel]}\tActual Class: {classes[testLabel[i]]}')
    testLoss, testAcc = tfModel.evaluate(testImage,  testLabel, verbose=2)
    print(f'\nTensorflow Test accuracy: {testAcc*100:.3f}%\n')

    plt.figure()
    imgRGB = testImage[i]
    plt.imshow(imgRGB)
    plt.xlabel(f'Predicted:{classes[predictedLabel]}')
    plt.title('TensorFlow Prediction')
    plt.grid(False)
    plt.show()

    return tf_infer_time


def pyTorchPredict(dataImages, dataLabels, i):

    dataset = torch.tensor(dataImages)

    all_data = []
    for n in range(len(dataset)):
        all_data.append([dataset[n], dataLabels[n]])

    test_data = all_data
    loss_fn = nn.CrossEntropyLoss()
 
    model.eval()

    x, y = test_data[i][0], test_data[i][1]
    x1 = x
    x = x.view(1, -1)

    with torch.no_grad():
        #test_loss, test_acc = test(test_dataloader, model, loss_fn)
        trch_start_time = time.time()
        pred = model(x.float())
        predicted, actual = classes[pred[0].argmax(0).item()], classes[y]
        pytorch_infer_time = time.time()-trch_start_time
        print('Pytorch Result\n')
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        #print(f'PyTorch Test Accuracy: {test_acc:.3f}%')
        img = x1.squeeze()
        plt.title('PyTorch Prediction')
        plt.imshow(img)
        plt.xlabel(f"Predicted: {classes[y]}")
        plt.show()
        return pytorch_infer_time


if __name__ == '__main__':

    i = 101
    classes = ['afiq', 'azureen', 'gavin', 'goke', 'inamul',
               'jincheng', 'mahmuda', 'numan', 'saseendran']

    print('\n\n_________________________Result___________________________')

    dataImages, dataLabels = importImageData(datasetFileName)
    tf_time = tensorflowPredict(dataImages, dataLabels, i)
    pt_time = pyTorchPredict(dataImages, dataLabels, i)
    print(
        f'\nTensorflow Predict Time:{tf_time}s\nPytorch Predict Time: {pt_time}s')

    print('\n\n_________________________End_______________________________')
