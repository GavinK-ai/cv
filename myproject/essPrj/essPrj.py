import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

list = []

def readImage(imgFile):

    if os.path.exists(imgFile):
        img = cv.imread(imgFile)
        return img
    else:
        print('File does not exist.')


def convert2Gry(rawImg):

    gry = cv.cvtColor(rawImg, cv.COLOR_BGR2GRAY)
    return gry


def apply_threshold(gryImg):

    ret, threshImg = cv.threshold(gryImg, 50, 255, cv.THRESH_BINARY)
    return ret, threshImg

def find_contours(threshImg):

    contours, hierarchy = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f"Detected contours: {len(contours)}")
    return contours, hierarchy

def find_shape(rawImg, contours):
    imgApproxPolyDP = rawImg.copy()
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approxPolyDP = cv.approxPolyDP(contour, epsilon, True)

        color = (0, 255, 255)
        thickness = 5
        # draw line
        for approx in approxPolyDP:
            cv.drawContours(imgApproxPolyDP, [approx], 0, color, thickness)
        color = (255, 0, 0)
        thickness = 5
        # draw points
        for approx in [approxPolyDP]:
            # draw points
            squeeze = np.squeeze(approx)
            print("contour:", approx.shape, squeeze.shape)
            for p in squeeze:
                pp = tuple(p.reshape(1, -1)[0])
                cv.circle(imgApproxPolyDP, pp, 10, color, -1)

        # determine shape
        verticeNumber = len(approxPolyDP)
        if verticeNumber == 3:
            vertice_shape = (verticeNumber, "Triangle")
        elif verticeNumber == 4:
            # get aspect ratio
            x, y, width, height = cv.boundingRect(approxPolyDP)
            aspectRatio = float(width) / height
            print(aspectRatio)
            if 0.90 < aspectRatio < 1.1:
                vertice_shape = (verticeNumber, "Square")
            else:
                vertice_shape = (verticeNumber, "Rectangle")
        elif verticeNumber == 5:
            vertice_shape = (verticeNumber, "Pentagon")
        elif verticeNumber == 6:
            vertice_shape = (verticeNumber, "Hexagon")
        elif verticeNumber == 7:
            vertice_shape = (verticeNumber, "Heptagon")
        elif verticeNumber == 8:
            vertice_shape = (verticeNumber, "Octagon")
        elif verticeNumber == 9:
            vertice_shape = (verticeNumber, "Nonagon")
        elif verticeNumber == 10:
            vertice_shape = (verticeNumber, "Decagon")
        else:
            vertice_shape = (verticeNumber, "Circle")

        # write shape
        # Compute the moment of contour:
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Get the position to draw:
        text = vertice_shape[1]
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        text_size = cv.getTextSize(text, fontFace, fontScale, thickness)[0]

        text_x = cX - text_size[0] / 2
        text_x = round(text_x)
        text_y = cY + text_size[1] / 2
        text_y = round(text_y)

        # Write the ordering of the shape on the center of shapes
        color = (0,0,0)
        result = cv.putText(
            imgApproxPolyDP, text, (text_x, text_y), fontFace, fontScale, color, thickness
        )
    
    return result

def sort_contours(img, sorted_size_shape_list):
    
    for i, (size, contour) in enumerate(sorted_size_shape_list):
        # Compute the moment of contour:
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Get the position to draw:
        text = str(i + 1)
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        thickness = 5
        text_size = cv.getTextSize(text, fontFace, fontScale, thickness)[0]

        text_x = cX - text_size[0] / 2
        text_x = round(text_x)
        text_y = cY + text_size[1] / 2
        text_y = round(text_y)

        # Write the ordering of the shape on the center of shapes
        color = (0, 0, 0)
        sort_result  = cv.putText(img, text, (text_x, text_y), fontFace, fontScale, color, thickness)
    
    return sort_result

def sizethenshape(rawImg,contours,sorted_size_shape_list):

    for i, (size, contour) in enumerate(sorted_size_shape_list):

        imgApproxPolyDP = rawImg.copy()
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approxPolyDP = cv.approxPolyDP(contour, epsilon, True)

            color = (0, 255, 255)
            thickness = 5
            # draw line
            for approx in approxPolyDP:
                cv.drawContours(imgApproxPolyDP, [approx], 0, color, thickness)
            color = (255, 0, 0)
            thickness = 5
            # draw points
            for approx in [approxPolyDP]:
                # draw points
                squeeze = np.squeeze(approx)
                print("contour:", approx.shape, squeeze.shape)
                for p in squeeze:
                    pp = tuple(p.reshape(1, -1)[0])
                    cv.circle(imgApproxPolyDP, pp, 10, color, -1)

            # determine shape
            verticeNumber = len(approxPolyDP)
            if verticeNumber == 3:
                vertice_shape = (verticeNumber, "Triangle")
            elif verticeNumber == 4:
                # get aspect ratio
                x, y, width, height = cv.boundingRect(approxPolyDP)
                aspectRatio = float(width) / height
                print(aspectRatio)
                if 0.90 < aspectRatio < 1.1:
                    vertice_shape = (verticeNumber, "Square")
                else:
                    vertice_shape = (verticeNumber, "Rectangle")
            elif verticeNumber == 5:
                vertice_shape = (verticeNumber, "Pentagon")
            elif verticeNumber == 6:
                vertice_shape = (verticeNumber, "Hexagon")
            elif verticeNumber == 7:
                vertice_shape = (verticeNumber, "Heptagon")
            elif verticeNumber == 8:
                vertice_shape = (verticeNumber, "Octagon")
            elif verticeNumber == 9:
                vertice_shape = (verticeNumber, "Nonagon")
            elif verticeNumber == 10:
                vertice_shape = (verticeNumber, "Decagon")
            else:
                vertice_shape = (verticeNumber, "Circle")

            # write shape
            # Compute the moment of contour:
            M = cv.moments(contour)

            # The center or centroid can be calculated as follows:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Get the position to draw:
            text = vertice_shape[1]+str(i)
            fontFace = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            text_size = cv.getTextSize(text, fontFace, fontScale, thickness)[0]

            text_x = cX - text_size[0] / 2
            text_x = round(text_x)
            text_y = cY + text_size[1] / 2
            text_y = round(text_y)

            # Write the ordering of the shape on the center of shapes
            color = (0,0,0)
            result = cv.putText(
                imgApproxPolyDP, text, (text_x, text_y), fontFace, fontScale, color, thickness
            )
        
        return result


    pass

def main():

    imgFile = 'shapes2.png'
    rawImg = readImage(imgFile)
    gryImg = convert2Gry(rawImg)
    ret, threshImg = apply_threshold(gryImg)
    contours, hierarchy = find_contours(threshImg)
    find_result = find_shape(rawImg, contours)

    contours_sizes = [cv.contourArea(contour) for contour in contours]
    size_shape_list = zip(contours_sizes, contours)
    sorted_size_shape_list = sorted(size_shape_list, key=lambda x: x[0])

    sortedContours = sort_contours(rawImg, sorted_size_shape_list)

    sortedSizethenShape = sizethenshape(rawImg,contours,sorted_size_shape_list)

    #find shapes and label
    # imgRGB = find_result[:,:,::-1]
    # plt.imshow(imgRGB)
    # plt.show()

    # gry2RGB = cv.cvtColor(threshImg, cv.COLOR_GRAY2RGB)
    # print(contours_sizes)

    #sort by size
    imgRGB = sortedSizethenShape[:,:,::-1]
    plt.imshow(imgRGB)
    plt.show()


    #sort by size then shape
    #find smallest size fist then sort for the shape


if __name__ == "__main__":

    main()

    
