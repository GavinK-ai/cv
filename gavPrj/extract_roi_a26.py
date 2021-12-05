import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


imageFile = 'Screenshot4.png'
filePath = 'assets/'+imageFile

if not os.path.exists('dataset'):
    os.mkdir('dataset')

folderPath = os.path.splitext(imageFile)[0]

folderPath = "dataset/"+folderPath

if not os.path.exists(folderPath):
    os.mkdir(folderPath)

pointList = []
pointList2 = []
pointList3 = []
pointList4 = []
pointListOfList = []

# Screenshot 1
pointList.append(((138, 108), (268, 257), 'numan'))
pointList.append(((515, 128), (661, 282), 'gavin'))
pointList.append(((906, 141), (1061, 301), 'afiq'))
pointList.append(((131, 355), (270, 513), 'goke'))
pointList.append(((501, 356), (704, 534), 'inamul'))
pointList.append(((906, 349), (1055, 508), 'azureen'))
pointList.append(((125, 584), (279, 751), 'mahmuda'))
pointList.append(((513, 557), (653, 712), 'saseendran'))
pointList.append(((925, 637), (1016, 737), 'jc'))
pointListOfList.append(pointList)

# Screenshot 2
pointList2.append(((319, 107), (479, 249), 'numan'))
pointList2.append(((688, 136), (848, 287), 'gavin'))
pointList2.append(((1120, 100), (1266, 269), 'saseendran'))
pointList2.append(((1101, 341), (1263, 485), 'inamul'))
pointList2.append(((1101, 556), (1240, 726), 'afiq'))
pointList2.append(((729, 564), (878, 720), 'goke'))
pointList2.append(((286, 538), (432, 652), 'jc'))
pointList2.append(((324, 346), (460, 523), 'mahmuda'))
pointList2.append(((694, 343), (840, 490), 'azureen'))
pointListOfList.append(pointList2)

pointList3.append(((353, 97), (527, 294), 'numan'))
pointList3.append(((877, 155), (1022, 358), 'gavin'))
# pointList2.append(((1120, 100), (1266, 269), 'saseendran'))
# pointList2.append(((1101, 341), (1263, 485), 'inamul'))
# pointList2.append(((1101, 556), (1240, 726), 'afiq'))
pointList3.append(((872, 440), (1040, 655), 'goke'))
# pointList2.append(((286, 538), (432, 652), 'jc'))
pointList3.append(((1381, 140), (1566, 368), 'mahmuda'))
pointList3.append(((346, 466), (531, 647), 'azureen'))
pointListOfList.append(pointList3)

pointList4.append(((375, 99), (550, 290), 'numan'))
pointList4.append(((877, 155), (1055, 359), 'gavin'))
# pointList2.append(((1120, 100), (1266, 269), 'saseendran'))
# pointList2.append(((1101, 341), (1263, 485), 'inamul'))
# pointList2.append(((1101, 556), (1240, 726), 'afiq'))
pointList4.append(((899, 436), (1072, 643), 'goke'))
# pointList2.append(((286, 538), (432, 652), 'jc'))
pointList4.append(((1403, 137), (1584, 363), 'mahmuda'))
pointList4.append(((362, 460), (559, 649), 'azureen'))
pointListOfList.append(pointList4)
count = 1

#for plist in pointListOfList:

for v in pointList4:

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

    saveFileName = folderPath+'/'+label+'.png'

    cv.imwrite(saveFileName, crop)

    a, b = 3, 3

    cropRGB = crop[:, :, ::-1]
    plt.subplot(a, b, count)
    plt.imshow(cropRGB)
    plt.title(label)

    count += 1

plt.show()
