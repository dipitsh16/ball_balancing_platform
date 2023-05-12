import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture(1)



cap.set(3,640)
cap.set(4,480)

class MotorPID():
    def __init__(self):
        self.sample_time=50
        self.max_values=255 #180
        self.min_values=0 #0

    def pid_box(self, new_point):
        
        self.setpoint = [new_point[0], new_point[1], 0]
        self.now = time.time()
        self.deltime = self.now - self.prev_time
        
        if (self.deltime>=self.sample_time):
            if(self.prev_time!=0):
                
                    self.error[0]= -(320 - self.setpoint[0])
                    self.error[1]= -(240 - self.setpoint[1])
                    self.error[2]=  (0 - self.setpoint[2])

                    
                
                    self.Kp = [15, 15, 21.0386]
                    self.Ki = [0,0,0]   
                    self.Kd = [34, 34, 45.6294]
                        
                    # Integral

                    self.integral_error[0] += self.error[0] * self.deltime
                    self.integral_error[1] += self.error[1] * self.deltime
                    self.integral_error[2] += self.error[2] * self.deltime

                    # Derivative


                    self.derivative_error[0] =  (self.error[0] - self.prev_error[0]) / (self.deltime)
                    self.derivative_error[1] =  (self.error[1] - self.prev_error[1]) / (self.deltime)
                    self.derivative_error[2] =  (0 - self.lastinput[2]) / self.deltime
                    

                    
                    # Optimizing parameters

                    self.out_roll     = self.Kp[0]*self.error[0] + self.Ki[0]*self.integral_error[0] + self.Kd[0]*self.derivative_error[0]
                    self.out_pitch    = self.Kp[1]*self.error[1] - self.Ki[1]*self.integral_error[1] + self.Kd[1]*self.derivative_error[1]
                    self.out_throttle = self.Kp[2]*self.error[2] + self.Ki[2]*self.integral_error[2] + self.Kd[2]*self.derivative_error[2]
                    
#                     print("output\n" + str(self.out_roll) + "  " + str(-self.out_pitch) )
                    
                    self.rcPitch    = int(255/2 + self.out_pitch)
                    self.rcRoll     = int(255/2 + self.out_roll)
                    self.rcThrottle = int(1500 + self.out_throttle)
                    
                    #Checking min and max threshold and updating on true
                    #Throttle Conditions
                    if self.rcThrottle>self.max_values:
                        self.rcThrottle=self.max_values
                    if self.rcThrottle<self.min_values:
                        self.rcThrottle=self.min_values     

                    #Pitch Conditions
                    if self.rcPitch>self.max_values:
                        self.rcPitch=self.max_values    
                    if self.rcPitch<self.min_values:
                        self.rcPitch=self.min_values

                    #Roll Conditionss
                    if self.cmd.rcRoll>self.max_values:
                        self.cmd.rcRoll=self.max_values
                    if self.cmd.rcRoll<self.min_values:
                        self.cmd.rcRoll=self.min_values

                    #Updating prev values for all axis
                    self.prev_error[:]=self.error[:]
                    self.lastinput[:] = [self.setpoint[0], self.setpoint[1], 0]



                    
if __name__=="__main__":                
  while True:
    pid = MotorPID()
    _,frame = cap.read()
    image = frame.copy()
    red = frame[:, :, 2]
    green = frame[:, :, 1]
    blue = frame[:, :, 0]
    key = cv2.waitKey(1)

    #Ball refinements
    red_only = np.int16(red)-np.int16(green)-np.int16(blue)
    red_only[red_only < 0] = 0
    red_only[red_only > 255] = 255
    red_only = np.uint8(red_only)
    cv2.imshow('red only', red_only)
    blur = cv2.GaussianBlur(red_only, (15, 15), 2)
    ret, bw_img = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", bw_img)

    #marking center of frame
    cv2.circle(image, (320, 240), 1, (0, 0, 255), -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray, (15, 15), 2)
    edged = cv2.Canny(blur2, 110, 115, L2gradient=True)
    ret, thresh = cv2.threshold(edged, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    cntrs, hierarchy = cv2.findContours(thresh, 1, 2)

    cx = 0
    cy = 0
    for cnt in cntrs:
        approx = cv2.approxPolyDP(cnt, 0.016 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio >= 1.0 and ratio <= 1.1:
                image = cv2.drawContours(image, [cnt], -1, (0, 255, 255), 3)
                N = cv2.moments(cnt)
                cx = int(N["m10"]/N["m00"])
                cy = int(N["m01"]/N["m00"])
                cv2.circle(image, (cx, cy), 3, (0, 0, 0), -1)
                cv2.putText(image, "plcnt", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # else:
            #     cv2.putText(image, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     image = cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
    #print(f"x: {cx} y: {cy}")
    cv2.imshow("frame.png", image)

    contours, hierarchies = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(bw_img.shape[:2], dtype='uint8')
    cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
    cv2.imshow("Contours.png", blank)
    cX = 0
    cY = 0
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #print(f"x: {cX} y: {cY}")

    pid.pid_box([cX, cY])

    errX = cx - cX
    errY = cy - cY
    print(f"x: {errX} y: {errY}")
    cv2.imshow("frame.png", image)
    cv2.imshow("image.png", frame)
    if key == ord("q"):
        break

# closing the window
cv2.destroyAllWindows()
cap.release()








    

