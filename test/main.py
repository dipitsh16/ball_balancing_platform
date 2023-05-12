import cv2
import numpy as np
import serial
import imutils

# reading the video
source = cv2.VideoCapture(1)

# running the loop
while True:
    ret, frame = source.read()
    # frame = cv2.resize(frame, (920, 600))
    # cv2.imshow("capture.png", capture)
    # bl = (20, 20)
    # tl = (20, 780)
    # br = (800, 20)
    # tr = (800, 780)
    #
    # cv2.circle(capture, bl, 3, (0, 0, 255), -1)
    # cv2.circle(capture, tl, 3, (0, 0, 255), -1)
    # cv2.circle(capture, br, 3, (0, 0, 255), -1)
    # cv2.circle(capture, tr, 3, (0, 0, 255), -1)
    #
    # pts1 = np.float32([tl, bl, tr, br])
    # pts2 = np.float32([[0, 0], [0, 820], [800, 0], [820, 800]])
    #
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # frame = cv2.warpPerspective(capture, matrix, (820, 800))
    image = frame.copy()
    red = frame[:, :, 2]
    green = frame[:, :, 1]
    blue = frame[:, :, 0]
    # cv2.imshow("Live", gray)
    key = cv2.waitKey(1)
    #Ball refinements
    red_only = np.int16(red)-np.int16(green)-np.int16(blue)
    red_only[red_only < 0] = 0
    red_only[red_only > 255] = 255
    red_only = np.uint8(red_only)
    cv2.imshow('red only', red_only)
    blur = cv2.GaussianBlur(red_only, (15, 15), 2)
    ret, bw_img = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    # converting to its binary form
    # bw = cv2.threshold(red_only, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", bw_img)

    #Platform refinements
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray, (15, 15), 2)
    edged = cv2.Canny(blur2, 130, 135, L2gradient=True)
    ret, thresh = cv2.threshold(edged, 20, 255, cv2.THRESH_BINARY)
    cntrs, hierarchy = cv2.findContours(thresh, 1, 2)
    # print("Number of contours detected:", len(cntrs))
    x1 = 0
    y1 = 0
    # print(cntrs)
    for cnt in cntrs:
        x1, y1 = cnt[0][0]
        # print(cnt)
        # print(cnt[0][0])
        approx = cv2.approxPolyDP(cnt, 0.016 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio >= 0.9 and ratio <= 1.1:
                image = cv2.drawContours(image, [cnt], -1, (0, 255, 255), 3)
                cv2.putText(image, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # else:
            #     cv2.putText(image, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     image = cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)

    # print(f"x: {x1} y: {y1}")
    #cv2.imwrite("thresh.png", bw_img)
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
    # print(f"x: {cX} y: {cY}")
    errX = x1 - cX
    errY = y1 - cY
    print(f"x: {errX} y: {errY}")
    cv2.imshow("frame.png", image)
    cv2.imshow("image.png", frame)
    if key == ord("q"):
        break

# closing the window
cv2.destroyAllWindows()
source.release()