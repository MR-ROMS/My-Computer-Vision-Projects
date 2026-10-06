import cv2 as cv
import numpy as np

img = cv.imread('Photos/boston.jpg')
img = cv.resize(img, (500, 500))
cv.imshow('Original', img)

# Edge Detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)
#Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
sobelx = np.uint8(np.absolute(sobelx))  
sobely = np.uint8(np.absolute(sobely))
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Combined Sobel', combined_sobel)
cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)

#Canny
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)




cv.waitKey(0)