# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import cv2
import dlib
import time
from scipy.spatial import distance


parser = argparse.ArgumentParser() #정수 목록을 받아 합계 또는 최댓값을 출력 파이썬 프로그램  #argumentparser객체를 생성하는 것
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')  #인자에 대한 정보를 add_argument()매서드를 호출
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')#argumentparser에게 명령행의 문자열을 객체로 변환하는 방법을 알려줌.
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args() #위의 정보는 저장되고 parse_args()가 호출될때 사용

#눈 찾기
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
###



BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              #["RElbow", "RWrist"],
              ["LShoulder", "LElbow"],
              #["LElbow", "LWrist"],
              #["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              #["LHip", "LKnee"], ["LKnee", "LAnkle"],
              ["Neck", "Nose"],
              #["Nose", "REye"],
              #["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
              ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else "v4l2src  device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=10/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1", cv.CAP_GSTREAMER)
#cap = cv.VideoCapture(args.input if args.input else "v4l2src  device=/dev/video0 ! video/x-raw,format=YUY2,width=368,height=368,framerate=10/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1", cv.CAP_GSTREAMER)




prevTime = 0  # 이전 시간을 저장할 변수


while (True):
    hasFrame, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ###

    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime

    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps

    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (125, 125, 125), 1)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (255, 0, 0), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)



    t, _ = net.getPerfProfile()
   # freq = cv.getTickFrequency() / 1000   #프레임 수 추출


##########################################################3


    #눈 찾기 코드
    # faces = hog_face_detector(gray)
    # for face in faces:
    #     face_landmarks = dlib_facelandmark(gray, face)
    #     leftEye = []
    #     rightEye = []
    #
    #     for n in range(36,42):
    #     	x = face_landmarks.part(n).x
    #     	y = face_landmarks.part(n).y
    #     	leftEye.append((x,y))
    #     	next_point = n+1
    #     	if n == 41:
    #     		next_point = 36
    #     	x2 = face_landmarks.part(next_point).x
    #     	y2 = face_landmarks.part(next_point).y
    #     	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
    #
    #     for n in range(42,48):
    #     	x = face_landmarks.part(n).x
    #     	y = face_landmarks.part(n).y
    #     	rightEye.append((x,y))
    #     	next_point = n+1
    #     	if n == 47:
    #     		next_point = 42
    #     	x2 = face_landmarks.part(next_point).x
    #     	y2 = face_landmarks.part(next_point).y
    #     	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)FU
    #
    #     left_ear = calculate_EAR(leftEye)
    #     right_ear = calculate_EAR(rightEye)
    #     # 눈동자 크기 계산
    #     EAR = (left_ear+right_ear)/2
    #     EAR = round(EAR,2)
    #     if EAR<0.18:
    #     	#cv2.putText(frame,"DROWSY",(20,100),
    #     	#	cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
    #     	cv2.putText(frame,"Sleep mode",(20,400), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)
        	#print("Drowsy")
        #print(EAR)

        #############################################################

     #load images
     #img1 = cv2.imread('./kpu.ac.kr.png')

    #put logo on top-left corner, create a ROI
    #rows, cols, channels = img1.shape
    #roi = frame[0:rows, 0:cols]

    #create mask of logo and create its inverse mask also
   #img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(img1gray, 23, 255, cv2.THRESH_BINARY)
   #mask_inv = cv2.bitwise_not(mask)

    #now black-out the area of log in ROI
   #cap_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    #Take only region of logo from logo image.
   #img1_fg = cv2.bitwise_and(img1, img1, mask=mask)

    #Put logo in ROI and modify the main image
   #dst = cv2.add(cap_bg, img1_fg)
   #frame[0:rows, 0:cols] = dst

    # gaussian Filtering
    blur = cv2.GaussianBlur(frame, (7,7),0)

    #lPF
    #blur = cv2.blur(frame, (5,5))

    #Median filtering (good for filtering)
    #blur = cv2.medianBlur(frame,5)

    #Bilateral Filtering (very slow,very Good)
    #blur = cv2.bilateralFilter(frame, 9, 75,75)

    # 프레임 확인
   # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    #본화면 출력
    cv.imshow('window', frame)

    # ESC버튼 누를 시 종료
    key = cv.waitKey(33)
    if key == 27:
        break


cap.release()
cv.destroyAllWindows()