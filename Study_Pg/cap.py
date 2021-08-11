import cv2

#capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap = cv2.VideoCapture(0) #CAP_V4L 에러 삭제
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set(cv2.CAP_PROP_FPS, 20)

while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    cv2.imshow("VideoFrame", frame)



cap.release()
cv2.destroyAllWindows()