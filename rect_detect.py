import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('white_plate.avi')

while(1):

    grabbed, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    #lower_red = np.array([161,155,84], dtype=np.uint8)
    #upper_red = np.array([179, 255, 255], dtype=np.uint8)

    # define range of green color in HSV
    lower_green = np.array([32, 90, 90], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    blur_mask = cv2.GaussianBlur(mask, (9, 9), 0)
    edged = cv2.Canny(blur_mask, 200, 500)
    edged = cv2.GaussianBlur(edged, (9,9),0)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)

        if len(approx) >= 1 and len(approx) <= 4:
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
    cv2.imshow("counturs",frame)

    #cv2.imshow("edge", edged)
   
    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
