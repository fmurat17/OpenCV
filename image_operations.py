import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

image_path = "itu.jpg"
image = cv2.imread(image_path) #BGR

print("Dimensions of the image -> {}".format(image.shape))
print("150,150 deki pixel değerleri -> {}".format(image[150,150]))


cv2.namedWindow("window name", cv2.WINDOW_FREERATIO)
cv2.imshow("window name", image)
cv2.waitKey()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Dimensions of the gray_image -> {}".format(gray_image.shape))
print("Pixel values of the gray scale image in [150,150] -> {}".format(gray_image[150,150]))

cv2.imshow("window name", gray_image)
cv2.waitKey()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------

onemsiz, thresh = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)

print("Dimensions of the thresholded image -> {}".format(thresh.shape))
print("Pixel values of the thresholded image in [150,150] -> {}".format(thresh[150,150]))

cv2.namedWindow("Original Image", cv2.WINDOW_FREERATIO)
cv2.namedWindow("Thresholded Image", cv2.WINDOW_FREERATIO)

cv2.imshow("Original Image", image)
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey()
cv2.destroyAllWindows()
#----------------------------------------------------------------------------

print(image.shape)

cv2.namedWindow("Original IMAGE", cv2.WINDOW_FREERATIO)
cv2.namedWindow("Blue", cv2.WINDOW_FREERATIO)
cv2.namedWindow("Green", cv2.WINDOW_FREERATIO)
cv2.namedWindow("Red", cv2.WINDOW_FREERATIO)

cv2.imshow("Original IMAGE", image)
cv2.waitKey()

blue = image.copy()
blue[:,:,1] = 0
blue[:,:,2] = 0

green = image.copy()
green[:,:,0] = 0
green[:,:,2] = 0

red = image.copy()
red[:,:,0] = 0
red[:,:,1] = 0


cv2.imshow("Blue", blue)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Green", green)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Red", red)
cv2.waitKey()
cv2.destroyAllWindows()
#-------------------------------------------------------------------

#HISTOGRAM#
cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
cv2.imshow("image", image)

# https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
#--------------------------------------------------------------------

cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
cv2.imshow("image", image)

plt.hist(image.flatten(), 256, [0,256])
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
#--------------------------------------------------------------------
print("image dimensions: {}".format(image.shape))

alt_degerler = (127, 127, 127)
ust_degerler = (255, 255, 255)


result = cv2.inRange(image, alt_degerler, ust_degerler)
print("result dimensions: {}".format(result.shape))

cv2.namedWindow("result", cv2.WINDOW_FREERATIO)
cv2.namedWindow("ORIGINAL", cv2.WINDOW_FREERATIO)
cv2.imshow("result", result)
cv2.imshow('ORIGINAL', image)

cv2.waitKey()
cv2.destroyAllWindows()


# Blue, Green, Red
alt_degerler2 = (  0,    0,   0)
ust_degerler2 = (255,  100, 100)


result2 = cv2.inRange(image, alt_degerler2, ust_degerler2)
print("result2 dimensions: {}".format(result2.shape))

cv2.namedWindow("result2", cv2.WINDOW_FREERATIO)
cv2.imshow("result2", result2)


cv2.waitKey()
cv2.destroyAllWindows()
#--------------------------------------------------------------------

image_path_split = 'Kodlar/images/renkler.png'

image_split = cv2.imread(image_path_split)

B = image_split[:, :, 0]
G = image_split[:, :, 1]
R = image_split[:, :, 2]

print(B.shape)
print(G.shape)
print(R.shape)

gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('ORIGINAL', cv2.WINDOW_FREERATIO)
cv2.namedWindow('B', cv2.WINDOW_FREERATIO)
cv2.namedWindow('G', cv2.WINDOW_FREERATIO)
cv2.namedWindow('R', cv2.WINDOW_FREERATIO)
cv2.namedWindow('grayscale', cv2.WINDOW_FREERATIO)

cv2.imshow('ORIGINAL', image_split)
cv2.waitKey()

cv2.imshow('grayscale', gray)
cv2.waitKey()

cv2.imshow('B', B)
cv2.waitKey()

cv2.imshow('G', G)
cv2.waitKey()

cv2.imshow('R', R)
cv2.waitKey()

cv2.destroyAllWindows()
#--------------------------------------------------------------------
