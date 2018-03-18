import cv2
import sys

# Read the image. The first command line argument is the image
#image = cv2.imread(sys.argv[1])
image = cv2.imread('ship.jpg')

cv2.imshow("Image", image)
cv2.waitKey(0)