import cv2 as cv

#Read Image
img = cv.imread('samples\data\home.jpg')

cv.imshow('Home',img)
cv.waitKey()