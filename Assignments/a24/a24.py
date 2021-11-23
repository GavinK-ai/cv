"""
Snapchat-based augmented reality OpenCV glasses overlay
"""

# Import required packages:
import cv2 as cv

# Load cascade classifiers for face and eyepair detection:
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eyepair_cascade = cv.CascadeClassifier("haarcascade_mcs_eyepair_big.xml")
nose_cascade = cv.CascadeClassifier("haarcascade_mcs_nose.xml")

# Load glasses image. The parameter -1 reads also de alpha channel (if exists)
# Open 'glasses.sgv' to see more glasses that can be used
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_glasses = cv.imread('assets/glasses4.png', -1)

# Rotate glasses image
img_glasses = cv.rotate(img_glasses, cv.ROTATE_180)
# Create the mask for the glasses:
img_glasses_mask = img_glasses[:, :, 3]
# cv.imshow("img glasses mask", img_glasses_mask)

# Convert glasses image to BGR (eliminate alpha channel):
img_glasses = img_glasses[:, :, 0:3]


# Load moustache image. The parameter -1 reads also de alpha channel
# Open 'moustaches.sgv' to see more moustaches that can be used
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_moustache = cv.imread('assets/moustache2.png', -1)

# Create the mask for the moustache:
img_moustache_mask = img_moustache[:, :, 3]
# cv.imshow("img moustache mask", img_moustache_mask)

# Convert moustache image to BGR (eliminate alpha channel):
img_moustache = img_moustache[:, :, 0:3]

# Rotate moustache image
img_moustache = cv.rotate(img_moustache, cv.ROTATE_180)

# You can use a test image to adjust the ROIS:
# test_face = cv.imread("../samples/data/face_test.png")
test_face = cv.imread("assets/numan_face.png")

# Create VideoCapture object to get images from the webcam:
video_capture = cv.VideoCapture(0)

while True:
    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Just for debugging purposes and to adjust the ROIS:
    # frame = test_face.copy()

    # Convert frame to grayscale:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using the function 'detectMultiScale()'
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face:
    for (x, y, w, h) in faces:
        # Draw a rectangle to see the detected face (debugging purposes):
        # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Create the ROIS based on the size of the detected face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect the eyepair inside the detected face:
        eyepairs = eyepair_cascade.detectMultiScale(roi_gray)

        # Detects a nose inside the detected face:
        noses = nose_cascade.detectMultiScale(roi_gray)

        # Iterate over the detected eyepairs (inside the face):
        for (ex, ey, ew, eh) in eyepairs:
            # Draw a rectangle to see the detected eyepair (debugging purposes):
            # cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 2)

            # Calculate the coordinates where the glasses will be placed:
            x1 = int(ex - ew / 10)
            x2 = int((ex + ew) + ew / 10)
            y1 = int(ey)
            y2 = int(ey + eh + eh / 2)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue

            # Draw a rectangle to see where the glasses will be placed (debugging purposes):
            # cv.rectangle(roi_color, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Calculate the width and height of the image with the glasses:
            img_glasses_res_width = int(x2 - x1)
            img_glasses_res_height = int(y2 - y1)

            # Resize the mask to be equal to the region were the glasses will be placed:
            mask = cv.resize(img_glasses_mask, (img_glasses_res_width, img_glasses_res_height))

            # Create the invert of the mask:
            mask_inv = cv.bitwise_not(mask)

            # Resize img_glasses to the desired (and previously calculated) size:
            img = cv.resize(img_glasses, (img_glasses_res_width, img_glasses_res_height))

            # Take ROI from the BGR image:
            roi = roi_color[y1:y2, x1:x2]

            # Create ROI background and ROI foreground:
            roi_bakground = cv.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv.bitwise_and(img, img, mask=mask)

            # Show both roi_bakground and roi_foreground (debugging purposes):
            # cv.imshow('roi_bakground', roi_bakground)
            # cv.imshow('roi_foreground', roi_foreground)

            # Add roi_bakground and roi_foreground to create the result:
            res = cv.add(roi_bakground, roi_foreground)

            # Set res into the color ROI:
            roi_color[y1:y2, x1:x2] = res

            break

        for (nx, ny, nw, nh) in noses:
            # Draw a rectangle to see the detected nose (debugging purposes):
            # cv.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

            # Calculate the coordinates where the moustache will be placed:
            x3 = int(nx - nw / 2)
            x4 = int(nx + nw / 2 + nw)
            y3 = int(ny + nh / 2 + nh / 8)
            y4 = int(ny + nh + nh / 4 + nh / 6)

            if x3 < 0 or x4 < 0 or x3 > w or y3 > h:
                continue

            # Draw a rectangle to see where the moustache will be placed (debugging purposes):
            # cv.rectangle(roi_color, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Calculate the width and height of the image with the moustache:
            img_moustache_res_width = int(x4 - x3)
            img_moustache_res_height = int(y4 - y3)

            # Resize the mask to be equal to the region were the glasses will be placed:
            mask = cv.resize(img_moustache_mask, (img_moustache_res_width, img_moustache_res_height))

            # Create the invert of the mask:
            mask_inv = cv.bitwise_not(mask)

            # Resize img_glasses to the desired (and previously calculated) size:
            img = cv.resize(img_moustache, (img_moustache_res_width, img_moustache_res_height))

            # Take ROI from the BGR image:
            roi = roi_color[y3:y4, x3:x4]

            # Create ROI background and ROI foreground:
            roi_bakground = cv.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv.bitwise_and(img, img, mask=mask)

            # Show both roi_bakground and roi_foreground (debugging purposes):
            # cv.imshow('roi_bakground', roi_bakground)
            # cv.imshow('roi_foreground', roi_foreground)

            # Add roi_bakground and roi_foreground to create the result:
            res = cv.add(roi_bakground, roi_foreground)

            # Set res into the color ROI:
            roi_color[y3:y4, x3:x4] = res

            break

    # Display the resulting frame
    cv.imshow('Snapchat-based OpenCV glasses filter', frame)
    

    # Press any key to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
video_capture.release()
cv.destroyAllWindows()