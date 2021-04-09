import numpy as np
import cv2
import imutils
import time

camera = cv2.VideoCapture('video.avi')
hasFrames, image = camera.read()
#cv2.imwrite('gate_image.jpg', image)
while(True):
	grabbed, img = camera.read()
	b = img.copy()
	#b = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#img[:,:,0] = 0
	#img[:,:,2] = 0
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

	#mask2 = cv2.bitwise_and(b, b, mask=mask)

	#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	#mask
	#mask = cv2.matchTemplate(mask, kernelImg, cv2.TM_CCOEFF)
	
	# Read image 
	# Find the edges in the image using canny detector
	
	
	#edges = cv2.Canny(mask, 10, 250)
	# Detect points that form a line
	#lines = cv2.HoughLinesP(edges, 4, np.pi/90, 200, minLineLength=40, maxLineGap=100)
	# Draw lines on the image
	#points = []
	#if not(lines is None):
	#	for e, line in enumerate(lines):
	#		x1, y1, x2, y2 = line[0]
	#		points.append((x1,y1))
	#		points.append((x2,y2))
	#		cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 3)
	
	#	points = np.array([points])
	#	print(points)
	#	cv2.fillPoly(b, points, True, 255)
	"""

	
	#mask2[:,:,0]= 0
	#mask2[:,:,2]= 0
	#mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
	#mask2 = cv2.threshold(mask2, 60, 255, cv2.THRESH_BINARY)[1]
	#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    	#cv2.imshow("green", mask);cv2.waitKey();cv2.destroyAllWindows()
	#mask = cv2.split(mask)
	#mask = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
	#mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)	
	#blur_mask = cv2.GaussianBlur(mask, (3, 3), 50)
	"""
	#edged = cv2.Canny(mask, 10, 250)
	#edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
	#edged = cv2.threshold(edged, 60, 255, cv2.THRESH_BINARY)[1]
	#edged = cv2.GaussianBlur(edged, (5,5),50)
	edged = mask.copy()
	
	

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
	                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#smooth = cv2.addWeighted( blur_mask, 1.5, mask, -0.5, 0)
	#img = cornerHarris_demo(200, smooth)
	#cv2.imshow("gate", edged);cv2.waitKey();cv2.destroyAllWindows()
	cv2.drawContours(edged, cnts, -1, (255,255,255), 4)
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		
		if (len(approx) >= 4 and len(approx) <= 10):
		#print(approx)
		#for d in approx:
			#for p in d:
				#print(p)
				#cv2.circle(edged, tuple(p), 10, (255,255,255),4)
			mask.fill(0)
			cv2.drawContours(mask, [approx], -1, (255, 0, 0), 4)
			cv2.fillPoly(mask, [approx], (255, 0, 0), 4)
	
	
	dst = cv2.cornerHarris(mask, 50,7,0.1)
	dst = cv2.dilate(dst,None)
	#print(dst)
	#print(mask2.dtype)
	#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	b[dst>0.2*dst.max()]=[0,0,255]
	#mask = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
	#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	#mask2 = np.float32(mask)

	mask2 = cv2.bitwise_and(b, b, mask=mask)
	#mask2 = np.float32(mask)

	#print (kernelImg.dtype, mask2.dtype)
	#mask2 = cv2.matchTemplate(mask2, kernelImg, cv2.TM_CCOEFF)
	#time.sleep(0.01)
	#mask2 = cv2.GaussianBlur(mask2, (5,5), 50)
	#img = cornerHarris_demo(200, edged)
	cv2.imshow("blur",hsv)
	cv2.imshow("mask",mask2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
camera.release()
cv2.waitKey()
cv2.destroyAllWindows()
