import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def create_canvas(height=500, width=500, bgColor=0):
    # create the image (canvas)
    canvas = np.zeros((height, width, 3), dtype='uint8')
    if bgColor != 0:
        canvas[:] = bgColor
    return canvas

def toRGB(img):
    return img[:,:,::-1]
    
def show_in_matplotlib(img, figsize=(6,8), title=None):
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # imgRGB = toRGB(img)
    plt.figure(figsize=figsize)
    
    plt.imshow(imgRGB)
    plt.title(title)
    plt.show()
