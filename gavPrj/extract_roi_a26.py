import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


filePath = 'assets\Screenshot1.png'

pointList = []

pointList.append(((138, 108), (268, 257), 'numan'))
pointList.append(((515, 128), (661, 282), 'gavin'))
pointList.append(((906, 141), (1061, 301), 'afiq'))
pointList.append(((131, 355), (270, 513), 'goke'))
pointList.append(((501, 356), (704, 534), 'inamul'))
pointList.append(((906, 349), (1055, 508), 'azureen'))
pointList.append(((125, 584), (279, 751), 'mahmuda'))
pointList.append(((513, 557), (653, 712), 'saseendran'))
pointList.append(((925, 637), (1016, 737), 'jc'))

count = 1

for v in pointList:

    (x1, y1), (x2, y2), label = v

    if y2 < y1:
        y = y2
        y2 = y1
        y1 = y

    if x2 < x1:
        x = x2
        x2 = x1
        x1 = x

    img = cv.imread(filePath)

    crop = img[y1:y2, x1:x2, :].copy()

    cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv.imwrite('dataset/Screenshot1/'+label+'.png', crop)

    a,b = 3,3

    cropRGB = crop[:, :, ::-1]
    plt.subplot(a,b,count)
    plt.imshow(cropRGB)
    plt.title(label)

    count+=1

plt.show()
    
