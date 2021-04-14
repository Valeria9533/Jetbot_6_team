import numpy as np
import cv2
import imutils
import time

camera = cv2.VideoCapture('video.avi')
hasFrames, image = camera.read()

def corners(img):
	
	X, Y = np.where(img > [0])
	coordinates = np.dstack((X, Y))
	return(coordinates)

while(True):
	grabbed, img = camera.read()
	b = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	blur_hsv = cv2.GaussianBlur(hsv, (3,3),50)
	thresh = cv2.threshold(b, 60, 255, cv2.THRESH_BINARY)[1]
	mask = cv2.inRange(blur_hsv,(32, 60, 80), (40, 255, 255) )
	
	kernel = np.ones((11, 11), np.float32)*255
	kernelImg = np.zeros([50,50,3], dtype = np.uint8)
	kernelImg.fill(255)
	mask = cv2.erode(mask,kernel,iterations=2)
	mask = cv2.dilate(mask, kernel, iterations = 17)
	mask = cv2.erode(mask,kernel,iterations=14)
	mask = cv2.dilate(mask, kernel, iterations = 10)
	mask = cv2.erode(mask,kernel,iterations=13)

	edged = mask.copy()
	
	

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
	                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cv2.drawContours(edged, cnts, -1, (255,255,255), 4)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		
		if (len(approx) >= 4 and len(approx) <= 10):
			mask.fill(0)
			cv2.drawContours(mask, [approx], -1, (255, 0, 0), 4)
			cv2.fillPoly(mask, [approx], (255, 0, 0), 4)
	
	
	dst = cv2.cornerHarris(mask, 50,7,0.1)
	dst = cv2.dilate(dst,None)
	
	
	b[dst>0.2*dst.max()]=[0,0,255]
	#mask = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
	#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	#mask2 = np.float32(mask)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
	#mask.fill(0)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	for i in centroids[1:]:
			cv2.rectangle(mask, (int(i[0]),int(i[1])), (int(i[0]+5), int(i[1]+5)),(255,0,0), 3)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	for i in centroids[1:]:
			cv2.rectangle(mask, (int(i[0]),int(i[1])), (int(i[0]+5), int(i[1]+5)),(255,0,0), 3)
	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	#corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	#print(corners)
	#print(img.shape)
	#mask2 = cv2.bitwise_and(b, b, mask=mask)
	#mask2 = np.float32(mask)

	#print (kernelImg.dtype, mask2.dtype)
	#mask2 = cv2.matchTemplate(mask2, kernelImg, cv2.TM_CCOEFF)
	#time.sleep(0.01)
	#mask2 = c		ev2.GaussianBlur(mask2, (5,5), 50)
	#img = cornerHarris_demo(200, edged)
	cv2.imshow("blur",hsv)
	cv2.imshow("mask",mask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
camera.release()
cv2.waitKey()
cv2.destroyAllWindows()

