import cv2 as cv

#Read Image
img = cv.imread('C:\SDK\Perantis\Perantis\cv\samples\data\home.jpg')

cv.imshow('Image',img)
cv.waitKey()