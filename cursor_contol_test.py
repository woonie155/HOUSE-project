import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui as pg
import time, os
import math


cap = cv2.VideoCapture(0)

camWidth, camHeight = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winWidth, winHeight = pg.size()
curX, curY = pg.position()
ratio = 0.4
print(camWidth, camHeight)


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)



seq = []
action_seq = []

def dist(x, x1, y, y1):
    res = (x-x1)*(x-x1)+(y-y1)*(y-y1)
    res=math.sqrt(res)
    return res

base_z_distance=0
base_x_position=0
base_y_position=0
z_distance=0
x_position=0
y_position=0
theta=0
time_=0
rate_x=0
rate_y=0

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
                curX = (((hand_landmarks.landmark[8].x - 0.5) / ratio) + 0.5) * winWidth
                curY = (((hand_landmarks.landmark[8].y - 0.5) / ratio) + 0.5) * winHeight
                pg.moveTo(int(curX), int(curY))

                pg.FAILSAFE=False
                mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break