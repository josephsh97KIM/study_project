import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
#import winsound as ws
from pygame import mixer
mixer.init()
alert=mixer.Sound('bell.wav')
import pickle
import time

# # 비프음
# def beepsound():
#     freq = 2000    # range : 37 ~ 32767
#     dur = 1000     # ms
#     ws.Beep(freq, dur) # winsound.Beep(frequency, duration)
# 눈 깜빡임 계산
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in dlib_facelandmark(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace= cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
# 윗 윕술
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append( landmarks[i])
    for i in range(61,64):
        top_lip_pts.append( landmarks[i])
    top_lip_all_pts = np.squeeze( np.asarray( top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])
#아랫 입술
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append( landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append( landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray( bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])
#하품
def mouth_open(image):
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

# 얼굴의 각 구격의 포인트들을 구분해 놓기
RIGHT_EYE_POINTS = list(range(36, 42)) # 오른쪽 눈
LEFT_EYE_POINTS = list(range(42, 48)) # 왼쪽 눈
MOUTH_OUTLINE_POINTS = list(range(48, 60)) # 입 바깥쪽
MOUTH_INNER_POINTS = list(range(60, 68)) # 입 안쪽

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 카메라 생성
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 하품 카운터
yawns = 0
yawn_status = False
# 눈 졸림
eye_flag = 0
start = time.time()
drowsy = 0  ##

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image_landmarks, lip_distance = mouth_open(frame)
    prev_yawn_status = yawn_status





    if lip_distance > 19:
        yawn_status = True
        time.sleep(0.5)
        cv2.putText(frame, "Subject is Yawning", (30,450), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        output_text = " Yawn Count: " + str(yawns + 1)
        cv2.putText(frame, output_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
    # If not lips distance is less than 25 then set yawn status to False
    else:
        yawn_status = False
    # Increasing yawn count if subject was yawning in previous frame as well
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        #눈깔 그리기
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        #prev_yawn_status = yawns_status

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        print(lip_distance)##


        EAR = 555*((left_ear + right_ear) / 2)
        #EAR = 500*round(EAR, 2)
        if EAR < 145:
            eye_flag += 1
            if eye_flag >= 24:  # 약 10초
                cv2.putText(frame, "alert", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                drowsy +=1 ##
                # print(beepsound()) # 경고음
                # alert.play()
        else:
            eye_flag = 0

        #print("Time:",time.time() - start)
        # print("flag :", eye_flag)
        # print(EAR)
        print("졸림",drowsy)  ##
        print("하품:", yawns) ##

    with open('test.pickle','wb') as f: ##
        pickle.dump(drowsy,f)
        pickle.dump(yawns,f)


    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow("study", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
