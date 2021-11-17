import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui as pg
import time, os
import math
import datetime as dt
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from win32api import GetSystemMetrics
import mouse

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume)) 


gesture = ['ALT_TAB', 'ALT_F4', 'FULL', 'SOUND_CONTROL']
seq_length = 30
model = load_model('models/cursor_model_t1.h5')
seq = []
gesture_seq = []

pg.FAILSAFE = False 

cap = cv2.VideoCapture(0)

camera_width, camera_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
display_width = GetSystemMetrics(0)
display_height = GetSystemMetrics(1)
ratio = 0.4
positionQueueSize = 5
cursor_position_list_x, cursor_position_list_y = mouse.get_position()
cursor_position_list_x = [cursor_position_list_x for i in range(positionQueueSize)]
cursor_position_list_y = [cursor_position_list_y for i in range(positionQueueSize)]
mouse_event_threshold = 0.04
mouse_event_threshold_angle = 0.5

is_pressed = [mouse.is_pressed('left'), mouse.is_pressed('middle'), mouse.is_pressed('right')]
is_detected = [False, False, False]
count = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5)

seq = []
gesture_seq = []

# def euclidean_distance_2d(x0, x1, y0, y1):
#     return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

# def euclidean_distance_3d(x0, x1, y0, y1, z0, z1):
#     return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

def arithmetic_mean(landmarks):
    return sum(landmark.x for landmark in landmarks) / len(landmarks),\
        sum(landmark.y for landmark in landmarks) / len(landmarks)

def position_mouse(landmarks):
    cursor_position_list_x.pop(0)
    cursor_position_list_y.pop(0)
    cursor_position_x, cursor_position_y = arithmetic_mean(list((landmarks[0].landmark[0], landmarks[0].landmark[17])))
    cursor_position_list_x.append((((cursor_position_x - 0.5) / ratio) + 0.5) * display_width)
    cursor_position_list_y.append((((cursor_position_y - 0.5) / ratio) + 0.5) * display_height)
    mouse.move(sum(cursor_position_list_x) / len(cursor_position_list_x), sum(cursor_position_list_y) / len(cursor_position_list_y))


def detect_bend2(landmarks):
    if (landmarks.landmark[8].y > landmarks.landmark[6].y):
        is_detected[0] = True
    else:
        is_detected[0] = False
    if (landmarks.landmark[12].y > landmarks.landmark[10].y):
        is_detected[1] = True
    else:
        is_detected[1] = False
    if (landmarks.landmark[16].y > landmarks.landmark[14].y):
        is_detected[2] = True
    else:
        is_detected[2] = False

def detect_bend(landmarks, threshold):
    v_out = np.array([
        landmarks.landmark[5].x - landmarks.landmark[8].x,
        landmarks.landmark[5].y - landmarks.landmark[8].y,
        landmarks.landmark[5].z - landmarks.landmark[8].z
    ])
    v_in = np.array([
        landmarks.landmark[0].x - landmarks.landmark[5].x,
        landmarks.landmark[0].y - landmarks.landmark[5].y,
        landmarks.landmark[0].z - landmarks.landmark[5].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold:
        is_detected[0] = True
    else:
        is_detected[0] = False
    v_out = np.array([
        landmarks.landmark[9].x - landmarks.landmark[12].x,
        landmarks.landmark[9].y - landmarks.landmark[12].y,
        landmarks.landmark[9].z - landmarks.landmark[12].z
    ])
    v_in = np.array([
        landmarks.landmark[0].x - landmarks.landmark[9].x,
        landmarks.landmark[0].y - landmarks.landmark[9].y,
        landmarks.landmark[0].z - landmarks.landmark[9].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold:
        is_detected[1] = True
    else:
        is_detected[1] = False
    v_out = np.array([
        landmarks.landmark[13].x - landmarks.landmark[16].x,
        landmarks.landmark[13].y - landmarks.landmark[16].y,
        landmarks.landmark[13].z - landmarks.landmark[16].z
    ])
    v_in = np.array([
        landmarks.landmark[0].x - landmarks.landmark[13].x,
        landmarks.landmark[0].y - landmarks.landmark[13].y,
        landmarks.landmark[0].z - landmarks.landmark[13].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold:
        is_detected[2] = True
    else:
        is_detected[2] = False

def click2(landmarks):
    detect_bend2(landmarks)
    if is_detected[0]:
        if not is_pressed[0]:
            mouse.press('left')
            is_pressed[0] = True
    else:
        if is_pressed[0]:
            mouse.release('left')
            is_pressed[0] = False

    if is_detected[1]:
        if not is_pressed[1]:
            mouse.press('middle')
            is_pressed[1] = True
    else:
        if is_pressed[1]:
            mouse.release('middle')
            is_pressed[1] = False

    if is_detected[2]:
        if not is_pressed[2]:
            mouse.press('right')
            is_pressed[2] = True
    else:
        if is_pressed[2]:
            mouse.release('right')
            is_pressed[2] = False

def click(landmarks, threshold):
    detect_bend(landmarks, threshold)
    if is_detected[0]:
        if not is_pressed[0]:
            mouse.press('left')
            is_pressed[0] = True
    else:
        if is_pressed[0]:
            mouse.release('left')
            is_pressed[0] = False

    if is_detected[1]:
        if not is_pressed[1]:
            mouse.press('middle')
            is_pressed[1] = True
    else:
        if is_pressed[1]:
            mouse.release('middle')
            is_pressed[1] = False

    if is_detected[2]:
        if not is_pressed[2]:
            mouse.press('right')
            is_pressed[2] = True
    else:
        if is_pressed[2]:
            mouse.release('right')
            is_pressed[2] = False
            
def arithMean(landmarks):
    sum_x = 0
    sum_y = 0
    for landmark in landmarks:
        sum_x += landmark.x
        sum_y += landmark.y
    return sum_x / 21, sum_y / 21

key_value= 0.08
key_flag = 0
#키보트 이벤트 가리기
def keybord_event_flag(landmarks): #4,8,12,16,20
    if(abs(landmarks[4].x - landmarks[16].x)<=key_value and abs(landmarks[4].x - landmarks[20].x)<=key_value
        and abs(landmarks[20].x - landmarks[16].x)<=key_value):
        if(abs(landmarks[4].y - landmarks[16].y)<=key_value and abs(landmarks[4].y - landmarks[20].y)<=key_value
        and abs(landmarks[20].y - landmarks[16].y)<=key_value):
            return 1
    if(landmarks[4].x>landmarks[2].x):
        return 1
    return 0
def volumn_value(landmarks):
    return (landmarks[5].x - landmarks[8].x)*100


start_time = dt.datetime.today().timestamp()

while cap.isOpened():
    count += 1
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    if result.multi_hand_landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
            if not keybord_event_flag(hand_landmarks.landmark):
                seq=[];
                if key_flag == 1: key_flag = 0;
                position_mouse(result.multi_hand_landmarks)
                #click(results.multi_hand_landmarks[0], mouse_event_threshold_angle)
                click2(result.multi_hand_landmarks[0])
                
                cv2.putText(
                    img, text=f'Hand Detected, {is_detected}, {is_pressed}', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(255, 255, 255), thickness=2)
                mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                
            else:    ############키보드 이벤트
                joint = np.zeros((21, 2))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y] 
                mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
                v = v2 - v1 # [20, 2] 
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 
                angle = np.degrees(angle) 
                add = np.concatenate([joint.flatten(), angle])
                seq.append(add)

                if len(seq) < seq_length: 
                    continue
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                acc = y_pred[i_pred]
                if acc < 0.98: 
                    continue
                key_event = gesture[i_pred]
                gesture_seq.append(key_event)
                if len(gesture_seq) < 2:
                    continue
                this_gesture = '?' 
                
                if gesture_seq[-1] == gesture_seq[-2]: 
                    if key_flag == 0:
                        this_gesture = key_event; res = hand_landmarks
                        if this_gesture == gesture[0]:
                            pg.hotkey('alt', 'tab')
                            print("알탭")
                            key_flag = 1;
                            
                        elif this_gesture == gesture[1]:
                            pg.hotkey('alt', 'F4')
                            print("알포")
                            key_flag = 1; 
                            
                        elif this_gesture == gesture[2]:
                            pg.press('f')
                            print("풀")
                            key_flag = 1; 
                            
                        else: 
                            v_value=volumn_value(hand_landmarks.landmark)
                            v_value= -((v_value+6) * 96/18)
                            v_value = min(0, v_value)
                            if v_value < -70:
                                v_value = -70.0
                            volume.SetMasterVolumeLevel(v_value, None)
                            print("볼륨")
                                        
                        cv2.putText(img, f'{this_gesture.upper()}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)                    

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break