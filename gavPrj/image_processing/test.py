import matplotlib.pyplot as plt

import cv2 as cv
from color_component import get_channel, remove_channel

img = cv.imread('color_img.png')

plt.subplot(3, 1, 1)
imgRGB = img[:, :, ::-1]
plt.imshow(imgRGB)

ch = 1
imgSingleChannel = get_channel(img, ch)
imgRGB = cv.cvtColor(imgSingleChannel, cv.COLOR_BGR2RGB)

plt.subplot(3, 1, 2)
plt.imshow(imgRGB)

imgChannelRemoved = remove_channel(img, ch)
imgRGB = imgChannelRemoved[:, :, ::-1]
plt.imshow(imgRGB)

plt.show()
