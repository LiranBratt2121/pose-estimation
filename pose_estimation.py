import cv2
import mediapipe as mp

from time import time

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pos = mp_pose.Pose()

p_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pos.process(img_rgb)
    
    c_time = time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)    

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape

            cx, cy = int(lm.x * w), int(lm.y * h)
            
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
            print(id, cx, cy)
            
    cv2.imshow('Image', img)
    
    cv2.waitKey(1)
    