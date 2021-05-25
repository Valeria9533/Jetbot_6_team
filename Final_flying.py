import numpy as np
import cv2
import imutils
import time
# import tellopy
# import av
from djitellopy import Tello

drone = Tello()

try:
    drone.connect()
    # drone.wait_for_connection(60.0)
except Exception as ex:
    print(ex)
    exit()

# drone.streamon()
# print("Hello hello hello")
# frame_read = drone.get_frame_read()

# camera = frame_read
# print("before")
# camera = av.open(drone.get_video_stream())
# print("after")
# time.sleep(5)
try:
    drone.takeoff()
except:
    drone.land()
    exit()
# time.sleep(10)

drone.streamon()

camera = drone.get_frame_read()
iterators = 0
imgCount = 0
IAmLost = 0
gates = 1
for n in range(gates):
    close = False
    tooManyWrong = 0
    while (True):

        # get_corners():
        ### Grabbing the video feed, "has frames" and "grabbed" check if
        ###there's a next frame, if there isn't, the feed will stop

        ### We'll have "img" and "image" for different purposes. "image" is the
        ### original video on top of which we draw, "img" is the one masked and
        ### used for getting contours to know what to draw.
        # try:
        print("Camera try")
        img = camera.frame
        image = camera.frame
        # cv2.imwrite("img.png", img)
        if img is None:
            print("none")
            continue
        # except:
        #    print("Camera fail")
        #    continue
        # hasFrames, image = camera.read()
        # except:
        #    drone.land()
        # grabbed, img = camera.read()
        # image = img
        ### Changing the frame into hsv colors and blurring it in various ways to smoothen
        """
        if (iterators == 0):
             drone.move_left(20)
             iterators += 1
             time.sleep(0.5)
        if (iterators == 1):
             drone.move_right(20)
             iterators += 1
             time.sleep(0.5)
        if (iterators == 2):
             drone.land()
             time.sleep(0.5)
        iterators += 2
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        blur_hsv = cv2.GaussianBlur(hsv, (1, 1), 0)
        blur_hsv = cv2.medianBlur(blur_hsv, 5)

        ### Create a white mask of the shape and blur the hell out of it

        mask = cv2.inRange(blur_hsv, (33, 90, 90), (80, 255, 255))

        blur_mask = cv2.GaussianBlur(mask, (1, 1), 0)
        blur_mask = cv2.medianBlur(blur_mask, 21)

        kernel = np.ones((11, 11), np.float32) * 255
        kernelImg = np.zeros([50, 50, 3], dtype=np.uint8)
        kernelImg.fill(255)

        mask = cv2.erode(blur_mask, kernel, iterations=3)

        mask = cv2.dilate(blur_mask, kernel, iterations=3)

        ###Resulting "img" used for contour counting
        img = blur_mask
        # cv2.imshow('showing', img)

        ### Gets edges and makes contours out of them and sorts them into a list
        edged = cv2.Canny(img, 10, 550)
        edged = cv2.medianBlur(edged, 1)
        # cv2.imshow('Filming', edged)

        cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # hull = np.array([[[5,5]],[[5,5]]])

        ### Empty list to be used later
        lista = np.array([])
        count = 0

        ###
        for c in cnts:

            ### approximate the contour and set minimum length fo rrecongised contours
            peri = cv2.arcLength(c, True)
            if peri >= 750:
                print("contours")
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)

                ### Collect long enough contours into the list
                lista = np.append(lista, approx).astype(int)
                count += len(approx)

            else:
                continue

            ### If there are between 4 and 10 corners, draw the contour on the "image"
            if len(approx) >= 4 and len(approx) <= 10:
                # cv2.imwrite("Test.png", image)
                cv2.drawContours(image, [approx], -1, (0, 0, 255), 5)

        try:
            ### This is "try", because all frames don't have contours and otherwise it would end the code
            print("try listing")
            lista = np.reshape(lista, (count, 2))

        except:
            continue

        mask2 = cv2.inRange(image, (0, 0, 250), (0, 0, 255))
        gray = mask2

        inline = False
        inlevel = False
        centered = False

        try:
            ### Draw the connecting contour (green) and use convex hull to surround it to get outermost edges and corners
            print("trying to get contours")
            cv2.drawContours(image, [lista], -1, (0, 255, 0), 5)
            hull = cv2.convexHull(lista)
            cv2.drawContours(image, [hull], -1, (255, 0, 0), 5)
            mask3 = cv2.inRange(image, (252, 0, 0), (255, 0, 0))

            corners = cv2.goodFeaturesToTrack(mask3, 4, 0.05, 110)
            corners = np.int0(corners)

            print("halfway contours")
            ### This get and draws he center of gate
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(mask3)
            mask3 = cv2.cvtColor(mask3, cv2.COLOR_GRAY2BGR)
            for i in centroids[1:]:
                cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[0] + 5), int(i[1] + 5)), (255, 0, 0), 3)
            ### And this gets the center of  image frame
            center_width = int(image.shape[1] / 2)
            center_height = int(image.shape[0] / 2)
            cv2.circle(image, (center_width, center_height), 10, (0, 0, 255), -1)
            print("end contours")
            ### Here we compare the two different centers to determine where to move
            # cv2.imshow("image", image)
            # cv2.imshow("img", img)

            if center_width - centroids[1][0] > 37:
                if close:
                   print('Fly Left')
                   drone.move_left(30)
                   time.sleep(0.1)
                   drone.move_right(20)
                   time.sleep(0.1)
                else:
                   drone.move_left(20)
                   time.sleep(0.1)

            elif center_width - centroids[1][0] < -37:
                if close:
                   print('Fly Right')
                   drone.move_right(30)
                   time.sleep(0.1)
                   drone.move_left(20)
                   time.sleep(0.1)
                else:
                   drone.move_right(20)
                   time.sleep(0.1)

            else:
                print('Stay in line')
                inline = True

            if center_height - centroids[1][1] > 95:
                print('Fly Up')
                drone.move_up(20)
                time.sleep(0.1)

            elif center_height - centroids[1][1] < 35:
                print('Fly Down')
                drone.move_down(20)
                time.sleep(0.1)
            else:
                print('Stay in Level')
                inlevel = True
                time.sleep(0.1)

            ### Draws yellow corners on "image"
            for i in corners:
                x, y = i.ravel()
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

            target = [0, 255, 255]
            X, Y = np.array(np.where(np.all(image == target, axis=2)))

            coordinates = np.array([])
            for c in range(0, 19, 5):
                coordinates = np.append(coordinates, X[c])
                coordinates = np.append(coordinates, Y[c])

            coordinates = np.reshape(coordinates, (4, 2))

        except:
            if IAmLost< 3:
                drone.move_forward(50)
                time.sleep(0.1)
                drone.rotate_counter_clockwise(15)
                time.sleep(0.1)
                IAmLost +=1
            else:
                drone.rotate_clockwise(90)
                time.sleep(0.1)
                IAmLost = 0
            print("didn't get contours")

            continue

            # cv2.imshow("gate2", image)

            # time.sleep(0.05)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

        # cv2.imshow("gate2", image)

        # cv2.imshow('Filming', img)
        # time.sleep(0.05)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
        bot_left = np.argmin(coordinates[2:4, 1]) + 2
        top_left = np.argmin(coordinates[0:2, 1])
        bot_right = np.argmax(coordinates[2:4, 1]) + 2
        top_right = np.argmax(coordinates[0:2, 1])
        gateHeigth = ((coordinates[bot_left][0] - coordinates[top_left][0])+(coordinates[bot_left][0] - coordinates[top_left][0]))/2
        gateWidth = (((coordinates[top_right][1])-(coordinates[top_left][1]))+((coordinates[bot_right][1])-(coordinates[bot_left][1])))/2
        gateShape = gateHeigth*0.5 < gateWidth < gateHeigth
        try:

            # print(coordinates[bot_left])
            # print(coordinates[2])
            if (coordinates[bot_left][0] - coordinates[top_left][0]) - (
                    coordinates[bot_right][0] - coordinates[top_right][0]) > 6:
                # print()
                # print('Rotate to Left ')
                # print()
                drone.rotate_counter_clockwise(10)
                time.sleep(0.1)

            if (coordinates[bot_right][0] - coordinates[top_right][0]) - (
                    coordinates[bot_left][0] - coordinates[top_left][0]) > 6:
                # print()
                # print('Rotate to Right ')
                # print()
                drone.rotate_clockwise(10)
                time.sleep(0.1)

            else:
                # print()
                # print('Centered')
                # print()
                centered = True

            if 50 < gateHeigth < 500 and gateShape:
                speed = 2.5 / gateHeigth * 4500
                #print(speed)
                if speed < 20:
                    speed = 20
                drone.move_forward(int(speed))
                time.sleep(0.1)
                close = False
            elif gateHeigth < 50:
                close = False
            elif gateShape == False:
                tooManyWrong += 1
            elif gateHeigth > 560 or tooManyWrong > 3:
                drone.move_back(20)
                time.sleep(0.1)
                if tooManyWrong > 4:
                    tooManyWrong = 0
                close = False 
            else:
                close = True

        except:
            print("rotate exception")
            continue

        try:
            leftRigthDist = (coordinates[bot_left][0] - coordinates[top_left][0]) > (
                        coordinates[bot_right][0] - coordinates[top_right][0])
            RigthLeftDist = (coordinates[bot_left][0] - coordinates[top_left][0]) < (
                    coordinates[bot_right][0] - coordinates[top_right][0])
            if ((leftRightDist and (coordinates[top_right][1] - coordinates[top_left][1]) < 200)):
                drone.move_right(25)
                print("moving right")
                time.sleep(0.1)
            elif ((RightLeftDist and (coordinates[top_right][1] - coordinates[top_left][1]) < 200)):
                drone.move_left(25)
                print("moving left")
                time.sleep(0.1)

        except:
            print("Angle exception")

        
	
        counter = 0        
        if gateShape == False:
            counter += 1
            if counter==3:
                drone.move_back(20)
                counter = 0
        try:
            imgName = "img" + str(imgCount)+".png"
            imageName = "image" + str(imgCount)+".png"
            print(imgName)
            print(imageName)
            cv2.imwrite(imgName, img)
            cv2.imwrite(imageName, image)
            imgCount += 1
        except:
            print("Fu")
        print(inline, inlevel, centered, close, gateShape)
        if inline and inlevel and centered and close and gateShape:
            #cv2.imwrite("img.png", img)
            #cv2.imwrite("image.png", image)
            drone.move_down(20)
            time.sleep(0.1)
            drone.move_forward(230)
            time.sleep(0.1)
            break

time.sleep(0.5)
drone.land()
# drone.quit()
camera.release()
# cv2.destroyAllWindows()
