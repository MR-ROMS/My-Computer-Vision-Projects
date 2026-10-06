import cv2 as cv
import numpy as np  

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #red color
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    red_mask = cv.inRange(hsv_frame, lower_red, upper_red)
    red = cv.bitwise_and(frame, frame, mask=red_mask)

    #Blue color
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    blue_mask = cv.inRange(hsv_frame, lower_blue, upper_blue)
    blue = cv.bitwise_and(frame, frame, mask=blue_mask)

    #Green color
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    green_mask = cv.inRange(hsv_frame, lower_green, upper_green)
    green = cv.bitwise_and(frame, frame, mask=green_mask)

    #Yellow color
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv.inRange(hsv_frame, lower_yellow, upper_yellow)
    yellow = cv.bitwise_and(frame, frame, mask=yellow_mask)
    
    #every color except white
    lower = np.array([0, 42, 0])
    upper= np.array([179, 255, 255])
    nOwhite_mask = cv.inRange(hsv_frame, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=nOwhite_mask)

    
    cv.imshow('frame', frame)
    #cv.imshow('red', red)
    #cv.imshow('blue', blue)
    #cv.imshow('green', green)
    #cv.imshow('yellow', yellow)
    cv.imshow('No white', result)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break   
