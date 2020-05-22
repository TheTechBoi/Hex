import numpy as np
import cv2


#NetworkTables.initialize(server='roborio-6025-frc.local') 
#table = NetworkTables.getTable("Vision") 


lower_color =  np.array([43, 70, 70])
upper_color =  np.array([73, 255, 255])

cap = cv2.VideoCapture('he.webm')

cam_angle = 60
cam_width = 1280
cam_center = cam_width/2
cam_centerAngle = cam_angle/2
ratio = cam_angle / cam_width


hexagon_w = 76.2
hexagon_h = 76.2/2
hexagon_percentage = int((hexagon_h/hexagon_w)*100)
minHexW = 20
minHexH = 10
hexagonVerification = False
hexEarlyVerification = False

x_difference = 0
angle_difference = 0

approx = []

with np.load('ilkCalibre.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]






def maxPixel(liste, index):
    number = 0
    i = 0
    for i in range(len(liste)):
        if number < liste[i][0][index]:
            number = liste[i][0][index]

            

    return number


def minPixel(liste, index):
    b = 0
    number2 = liste[b][0][index]
    for b in range(len(liste)):
        if number2 > liste[b][0][index]:
            number2 = liste[b][0][index]
    return number2

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img




imgp = np.zeros((8,2), np.float32)
objp = np.zeros((8,3), np.float32)

axis = np.float32([[1,0,0], [0,1,0], [0,0,-1],[0,0,0]]).reshape(-1,3)

p_one  = ( -20,   8.8, 0) 
p_two  = (-16.8,   8.8, 0)
p_thr  = (-13.5, 2.3, 0)
p_four = (-6.5, 2.3, 0)
p_five = (-3.2,   8.8, 0)
p_six  = (   0,   8.8, 0)
p_sev  = (-5,   0, 0)
p_eig  = (-15,   0, 0)

objp[0] = p_one
objp[1] = p_two
objp[2] = p_thr
objp[3] = p_four
objp[4] = p_five
objp[5] = p_six
objp[6] = p_sev
objp[7] = p_eig


x = 0
y = 0

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while cap.isOpened():
    _, frame = cap.read()

##    h,  w = frame.shape[:2]
##    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
##    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((15,15),np.float32)/225
    
    smoothed = cv2.filter2D(hsv,-1,kernel)

    hsv_blur = cv2.medianBlur(smoothed,15)
    

    
    mask = cv2.inRange(hsv_blur, lower_color, upper_color)

    mask = cv2.erode(mask,kernel,iterations =2)
    mask = cv2.dilate(mask,kernel, iterations=2) 

    res = cv2.bitwise_and(frame,frame, mask = mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
##    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    _, thrash = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:

            for c in contours:
                if 1000000 > cv2.contourArea(c) > 500 :
                    approx = cv2.approxPolyDP(c, 0.03* cv2.arcLength(c, True), True)
                    approx = cv2.approxPolyDP(approx, 0.01* cv2.arcLength(c, True), True)
                    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                    x_ = approx.ravel()[0]
                    y_ = approx.ravel()[1] - 7
                    if len(approx) == 8:
                        hexEarlyVerification = True
                        break
                    else:
                        hexEarlyVerification = False

        
            if(hexEarlyVerification == True):
                cv2.putText(frame, "Lol", (x_, y_), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                print(approx)
                print('\n')
                maxX = maxPixel(c,0)
                minX = minPixel(c,0)
                maxY = maxPixel(c,1)
                minY = minPixel(c,1)
            
                shapeW = maxX - minX
                shapeH = maxY - minY

                if shapeW >= minHexW and shapeH >= minHexH:
                    print('W is:')
                    print(shapeW)
                    print('H is:')
                    print(shapeH)
                    shape_percentage = int((shapeH/shapeW)*100)

                    if (hexagon_percentage - 10) <= shape_percentage <= (hexagon_percentage + 10):
                        hexagonVerification = True
                    else:
                        hexagonVerification = False


                if hexagonVerification == True:
                    
                    x = minX + int(shapeW/2) 
                    y = minY + int(shapeH/2)

                    cv2.circle(frame,(x,y), 6, (255, 0, 0), -1)
                    
                    x_difference = cam_center - x
                    angle_difference = 0-(x_difference*ratio)

                    imgp[0] = tuple(approx[0][0])
                    imgp[1] = tuple(approx[1][0])
                    imgp[2] = tuple(approx[2][0]) 
                    imgp[3] = tuple(approx[3][0])
                    imgp[4] = tuple(approx[4][0])
                    imgp[5] = tuple(approx[5][0])
                    imgp[6] = tuple(approx[6][0])
                    imgp[7] = tuple(approx[7][0]) 
                    

                    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, imgp, mtx, dist)
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    try:
                        frame = draw(frame,imgp,imgpts)
                    except:
                        print('\n\n\n ASSJFNALJSBFLJASBLFJBALJSNDL ASNFOJNASLNDLASNDLJNLJASNDLA JSLD \n\n\n')

                        
                    print(imgpts)
                else:
                    x = 0
                    y = 0
                    angle_difference = 0

    else:
        x = 0
        y = 0
        angle_difference = 0

    print("x : ")
    print(x)
    print("y : ")
    print(y) 
    print("Angle : ")
    print(angle_difference)
   

    print('\n\n\n')

    cv2.imshow("frame", frame )
    cv2.imshow("res", res )

    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
    
cv2.destroyAllWindows()
cap.release()
