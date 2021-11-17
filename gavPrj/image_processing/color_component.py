import cv2 as cv
from matplotlib.pyplot import title
from utilities import show_in_matplotlib




def get_channel(img, channel):
    b = img[:,:,channel]
    #g = img[:,:,1]
    #r = img[:,:,2]

    #show_in_matplotlib(b)
    return b

def remove_channel(img, channel):

    b = img[:,:,channel]
    g = img[:,:,1]
    r = img[:,:,2]

    if channel == 0:
        b[:] = 0
    elif channel == 1:
        g[:] = 0
    else:
        r[:] = 0

    img_merged = cv.merge((b,g,r))
    #show_in_matplotlib(img_merged)

    return img_merged

if __name__ == "__main__":
    
    img = cv.imread('color_img.png')
    show_in_matplotlib(img, title="Original")

    ch=1
    b = get_channel(img, ch)
    show_in_matplotlib(b, title=f"Channel {ch} only")

    img_merged = remove_channel(img, ch)
    show_in_matplotlib(img_merged, title=f"Channel {ch} removed")
