import copy
import glob
import os
import time
from pathlib import Path
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
from data_process import (add_noise, create_data, flip, greyscale,
                          normalization, rotate, process)



train_path = 'dataset/training_data_aug'
test_path = 'dataset/testing_data'

# create dataset
df_train = process(train_path)
df_test = process(test_path)

# add augmented train data


# define parameters
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_SEED = 42

# generating images
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_image = train_generator.flow_from_dataframe(dataframe=df_train,
                                                  x_col='filepaths',
                                                  y_col='labels',
                                                  target_size=(224, 224),
                                                  batch_size=BATCH_SIZE,
                                                  subset='training',
                                                  random_seed=RANDOM_SEED)

test_image = test_generator.flow_from_dataframe(dataframe=df_test,
                                                x_col='filepaths',
                                                y_col='labels',
                                                target_size=(224, 224),
                                                batch_size=BATCH_SIZE,
                                                random_seed=RANDOM_SEED
                                                )

pretrained_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)
# pretrained_model = ResNet50V2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet',
#     pooling='max',
# )
pretrained_model.trainable = False  # We don't want to train again the resnet

inputs = pretrained_model.input

x = Dense(120, activation='relu')(pretrained_model.output)
x = Dense(120, activation='relu')(x)  # adding some custom layers of our coice
x = Dense(120, activation='relu')(x)
x = Dense(120, activation='relu')(x)

outputs = Dense(2, activation='sigmoid')(x)
# output choice
model = Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

mo_fit = model.fit(train_image, epochs=EPOCHS)

train_acc = mo_fit.history


pd.DataFrame(mo_fit.history)[['accuracy']].plot()
plt.title("Train Accuracy")
plt.show()
#plt.imsave('Training Accuracy.png')

pd.DataFrame(mo_fit.history)[['loss']].plot()
plt.title("Train Loss")
plt.show()
#plt.imsave('Training Loss.png')


print('Test Accuracy')
eval_result = model.evaluate(test_image)


savedFile = f'resnet_aug/cv_image_resnet_{(eval_result[1]*100):.0f}.pt'
model.save(savedFile)
print("Export Path = "+savedFile)

