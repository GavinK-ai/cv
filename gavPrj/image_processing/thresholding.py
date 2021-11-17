


def binary(imgGray, thresh=50):
    import cv2 as cv

    ret, thresh_img = cv.threshold(imgGray, thresh, 255, cv.THRESH_BINARY)
    # plt.imshow(thresh_img)
    # plt.title('thresh=0, maxval=255')

    # plt.show()
    return thresh_img

if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    import cv2 as cv
    import sys

    filePath = 'gray_boxes.png'

    if len(sys.argv)>1:
        filePath = sys.argv[1]
        
    #load file
    img = cv.imread(filePath)

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, imgThreshed = binary(imgGray)

    plt.subplot(3,1,1)
    imgRGB = img[:,:,::-1]
    plt.imshow(imgRGB)
    plt.title("Orignal")

    plt.subplot(3,1,2)
    imgGray = img[:,:,::-1]
    plt.imshow(imgGray)
    plt.title("Greyscale")

    plt.subplot(3,1,3)
    imgRGB = cv.cvtColor(imgThreshed, cv.COLOR_GRAY2RGB)
    plt.imshow(imgRGB)
    plt.title(f"Thres-{ret}, maxVal=255")

    plt.show()