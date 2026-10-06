import cv2 as cv
import numpy as np  
import matplotlib.pyplot as plt


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.medianBlur(frame, 11)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   

    lower_red= np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    res = cv.bitwise_and(frame, frame, mask=mask)

    #blur = cv.GaussianBlur(gray, (5 , 5), cv.BORDER_DEFAULT)
    #canny = cv.Canny(blur, 125, 175)
    
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 1)
    
    for contour in contours:
        area = cv.contourArea(contour)
        if 500 < area < 5000:
            x, y, w, h = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


    
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
