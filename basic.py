import cv2 as cv
import numpy as np


img = cv.imread('Photos/00000_00024.ppm')

cv.imshow('image', img)
cv.waitKey(0)