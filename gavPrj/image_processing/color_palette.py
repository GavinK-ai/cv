import random
import cv2 as cv

from utilities import create_canvas, show_in_matplotlib


def default_palette():

    img = create_canvas(100, 200, (255, 255, 255))

    s = 40
    p = 5  # padding
    r = 0
    (l, t) = (r+p, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (255, 0, 0), -1)

    (l, t) = (r+p, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (0, 255, 0), -1)

    (l, t) = (r+p, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (0, 0, 255), -1)

    (l, t) = (r+p, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (0, 0, 0), -1)

    r = 0
    p = 50
    (l, t) = (r+5, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (255, 255, 0), -1)

    (l, t) = (r+5, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (255, 0, 255), -1)

    (l, t) = (r+5, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (0, 255, 255), -1)

    (l, t) = (r+5, p)
    (r, b) = (l+s, t+s)
    cv.rectangle(img, (l, t), (r, b), (255, 255, 255), -1)

    cv.imwrite('color_img.png', img)

    show_in_matplotlib(img, None)


def create_palette():

    import cv2 as cv

    squareSize = 50
    padding = 5
    nrows = 4
    ncols = 4

    # silver or gray background
    height = (nrows * squareSize) + ((nrows+1)*padding)
    width = (ncols * squareSize) + ((ncols+1)*padding)
    bgColor = (155, 155, 155)

    img = create_canvas(height, width, bgColor)

    b = 0
    # draw each row of square color
    for i in range(nrows):
        # start each row at point 0 on the right
        r = 0
        # start each row at the top at point bottom + padding
        t = b + padding
        # bottom of the square
        b = t + squareSize

        # draw each square per column on a row
        for j in range(ncols):
            l = r + padding
            r = l + squareSize
            color = random.choices([0, 150, 255])[0], random.choices(
                [0, 150, 255])[0], random.choices([0, 150, 255])[0]
            # color = random.randint(0,255), random.randint(0,255), random.randint(0,255)
            cv.rectangle(img, (l, t), (r, b), color=color, thickness=-1)

    # save the image
    cv.imwrite(f'color_img_{nrows}x{ncols}.png', img)

    show_in_matplotlib(img, None)

if __name__ == "__main__":

    default_palette()

    create_palette()