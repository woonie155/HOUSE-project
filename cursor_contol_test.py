import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui as pg
import time, os
import math

actions = ['ALT_F4', 'ALT_TAB', 'ENTER']
seq_length = 30
model = load_model('models/model.h5')
seq = []
action_seq = []

pg.FAILSAFE = False 

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

def eucDist2d(x0, x1, y0, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def arithMean(landmarks):
    sum_x = 0
    sum_y = 0
    for landmark in landmarks:
        sum_x += landmark.x
        sum_y += landmark.y
    return sum_x / 21, sum_y / 21

#키보트 이벤트 가리기
def keybord_event_flag(landmarks): #4,8,12,16,20
    if(abs(landmarks[4].x - landmarks[16].x)<=0.3 and abs(landmarks[4].x - landmarks[20].x)<=0.3
        and abs(landmarks[20].x - landmarks[16].x)<=0.3):
        if(abs(landmarks[4].y - landmarks[16].y)<=0.3 and abs(landmarks[4].y - landmarks[20].y)<=0.3
        and abs(landmarks[20].y - landmarks[16].y)<=0.3):
            return 1
    return 0


while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
            joint = np.zeros((21, 2))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y]

            if not keybord_event_flag(hand_landmarks.landmark):
                curX, curY = arithMean(hand_landmarks.landmark)
                curX = (((curX - 0.5) / ratio) + 0.5) * winWidth
                curY = (((curY - 0.5) / ratio) + 0.5) * winHeight
                pg.moveTo(int(curX), int(curY))

                cv2.putText(img, text='Hand Detected', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                    color=(0, 0, 0), thickness=2)
                mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                if eucDist2d(hand_landmarks.landmark[8].x,
                            hand_landmarks.landmark[4].x,
                            hand_landmarks.landmark[8].y,
                            hand_landmarks.landmark[4].y) < 0.01:
                    pg.click()

                
            else:            ############키보드 이벤트
                mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
                v = v2 - v1 # [20, 2] #각 x,y,z의 마디마다의 차이
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n', #행별로 벡터연산
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                angle = np.degrees(angle) # Convert radian to degree
                d = np.concatenate([joint.flatten(), angle])
                seq.append(d)
                if len(seq) < seq_length: #계속판별
                    continue
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                if conf < 0.95: #90넘어야지만 저장
                    continue
                action = actions[i_pred]
                action_seq.append(action)
                if len(action_seq) < 3:
                    continue
                this_action = '?' #판단안되는경우
                
                if action_seq[-1] == action_seq[-2] == action_seq[-3]: 
                    this_action = action
                    res= hand_landmarks
                    cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break