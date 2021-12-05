import sys
import cv2 as cv

print(f'Argument array \n{sys.argv}')
print('>>> 1st argument: ', sys.argv[0])
print('>>> 2nd argument: ', sys.argv[1])

filePath = sys.argv[1]

# Read video from path
capture = cv.VideoCapture(filePath)

# Check if connected
if capture.isOpened() is False:
    print("Error opening camera 0")
    exit()

# Get and print video information
frame_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv.CAP_PROP_FPS)
frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
frame_delay = frame_count/fps

print(f'Frame per seconds: {fps:.2f} fps')
print(f'Video W x H: {frame_width} x {frame_height} px')
print(f'Frame count: {frame_count}')
print(f'Frame delay: {frame_delay:.2f} ms')

while capture.isOpened():
    # Capture frames, if read correctly ret is True
    ret, frame = capture.read()

    # If no frame is returned, stop the loop
    if not ret:
        print("Frame stopped.")
        break

    # Show video
    cv.imshow("Camera frame", frame)

    # Set frame delay
    k = cv.waitKey(int(frame_delay))

    # Check if key is q then exit
    if k == ord("q"):
        break

# Close the video
capture.release()
cv.destroyAllWindows()
