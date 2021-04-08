import numpy as np
import cv2
import imutils


camera = cv2.VideoCapture('gate_video2.avi')
hasFrames, image = camera.read()
#cv2.imwrite('gate_image.jpg', image)
while(True):
	grabbed, img = camera.read()
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	blur_hsv = cv2.GaussianBlur(hsv, (5,5),0)
	mask = cv2.inRange(blur_hsv,(32, 90, 90), (85, 255, 255) )
	#cv2.imshow("green", mask);cv2.waitKey();cv2.destroyAllWindows()

	blur_mask = cv2.GaussianBlur(mask, (9, 9), 0)
	edged = cv2.Canny(blur_mask, 50, 150)
	edged = cv2.GaussianBlur(edged, (9,9),0)
	
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	#cv2.imshow("gate", edged);cv2.waitKey();cv2.destroyAllWindows()


	for c in cnts:
	    # approximate the contour
	    peri = cv2.arcLength(c, True)
	    approx = cv2.approxPolyDP(c, 0.1 * peri, True)
	
	    if (len(approx) >= 4 and len(approx) <= 12):
	        cv2.drawContours(img, [approx], -1, (0, 0, 255), 4)
	cv2.imshow("gate2",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
camera.release()
cv2.waitKey()
cv2.destroyAllWindows()



