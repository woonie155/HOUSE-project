import cv2
import math
import mediapipe as mp
import mouse
import datetime as dt
from win32api import GetSystemMetrics
import numpy as np

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camera_width, camera_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
display_width = GetSystemMetrics(0)
display_height = GetSystemMetrics(1)
cursor_position_list_x, cursor_position_list_y = mouse.get_position()
cursor_position_list_x = [cursor_position_list_x for i in range(5)]
cursor_position_list_y = [cursor_position_list_y for i in range(5)]
ratio = 0.4
mouse_event_threshold = 0.04
mouse_event_threshold_angle = 0.5

is_pressed = [mouse.is_pressed('left'), mouse.is_pressed('middle'), mouse.is_pressed('right')]
is_detected = [False, False, False]

print(display_width, display_height)

count = 0


def euclidean_distance_2d(x0, x1, y0, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def euclidean_distance_3d(x0, x1, y0, y1, z0, z1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

def arithmetic_mean(landmarks):
    return sum(landmark.x for landmark in landmarks) / len(landmarks),\
        sum(landmark.y for landmark in landmarks) / len(landmarks)

def detect_bend(landmarks, threshold, finger_num):
    if finger_num == 2:
        v_out = np.array([
            landmarks[0].landmark[5].x - landmarks[0].landmark[8].x,
            landmarks[0].landmark[5].y - landmarks[0].landmark[8].y,
            landmarks[0].landmark[5].z - landmarks[0].landmark[8].z
        ])
        v_in = np.array([
            landmarks[0].landmark[0].x - landmarks[0].landmark[5].x,
            landmarks[0].landmark[0].y - landmarks[0].landmark[5].y,
            landmarks[0].landmark[0].z - landmarks[0].landmark[5].z
        ])
        v_out = v_out / np.linalg.norm(v_out)
        v_in = v_in / np.linalg.norm(v_in)
        # print('2', v_in, v_out)
        if np.dot(v_out, v_in) < threshold:
            return True
        else:
            return False
    elif finger_num == 3:
        v_out = np.array([
            landmarks[0].landmark[9].x - landmarks[0].landmark[12].x,
            landmarks[0].landmark[9].y - landmarks[0].landmark[12].y,
            landmarks[0].landmark[9].z - landmarks[0].landmark[12].z
        ])
        v_in = np.array([
            landmarks[0].landmark[0].x - landmarks[0].landmark[9].x,
            landmarks[0].landmark[0].y - landmarks[0].landmark[9].y,
            landmarks[0].landmark[0].z - landmarks[0].landmark[9].z
        ])
        v_out = v_out / np.linalg.norm(v_out)
        v_in = v_in / np.linalg.norm(v_in)
        # print('3', v_in, v_out)
        if np.dot(v_out, v_in) < threshold:
            return True
        else:
            return False
    elif finger_num == 4:
        v_out = np.array([
            landmarks[0].landmark[13].x - landmarks[0].landmark[16].x,
            landmarks[0].landmark[13].y - landmarks[0].landmark[16].y,
            landmarks[0].landmark[13].z - landmarks[0].landmark[16].z
        ])
        v_in = np.array([
            landmarks[0].landmark[0].x - landmarks[0].landmark[13].x,
            landmarks[0].landmark[0].y - landmarks[0].landmark[13].y,
            landmarks[0].landmark[0].z - landmarks[0].landmark[13].z
        ])
        v_out = v_out / np.linalg.norm(v_out)
        v_in = v_in / np.linalg.norm(v_in)
        # print('4', v_in, v_out)
        if np.dot(v_out, v_in) < threshold:
            return True
        else:
            return False

def position_mouse(landmarks):
    cursor_position_list_x.pop(0)
    cursor_position_list_y.pop(0)
    cursor_position_x, cursor_position_y = arithmetic_mean(list((landmarks[0].landmark[0], landmarks[0].landmark[17])))
    cursor_position_list_x.append((((cursor_position_x - 0.5) / ratio) + 0.5) * display_width)
    cursor_position_list_y.append((((cursor_position_y - 0.5) / ratio) + 0.5) * display_height)
    mouse.move(sum(cursor_position_list_x) / len(cursor_position_list_x), sum(cursor_position_list_y) / len(cursor_position_list_y))

def click(landmarks, threshold):
    if detect_bend(landmarks, threshold, 2):
        is_detected[0] = True
        if not mouse.is_pressed('left'):
            mouse.press('left')
            is_pressed[0] = mouse.is_pressed('left')
    else:
        is_detected[0] = False
        if mouse.is_pressed('left'):
            mouse.release('left')
            is_pressed[0] = mouse.is_pressed('left')

    if detect_bend(landmarks, threshold, 3):
        is_detected[1] = True
        if not mouse.is_pressed('middle'):
            mouse.press('middle')
            is_pressed[1] = mouse.is_pressed('middle')
    else:
        is_detected[1] = False
        if mouse.is_pressed('middle'):
            mouse.release('middle')
            is_pressed[1] = mouse.is_pressed('middle')
            
    if detect_bend(landmarks, threshold, 4):
        is_detected[2] = True
        if not mouse.is_pressed('right'):
            mouse.press('right')
            is_pressed[2] = mouse.is_pressed('right')
    else:
        is_detected[2] = False
        if mouse.is_pressed('right'):
            mouse.release('right')
            is_pressed[2] = mouse.is_pressed('right')


hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)

start_time = dt.datetime.today().timestamp()

while cap.isOpened():
    count += 1
    isSucceeded, image = cap.read()
    if not isSucceeded:
        continue
    image = cv2.cvtColor(cv2.flip(image, 1), 1)
    results = hands.process(image)

    # print(mouse.is_pressed('left'))

    if results.multi_hand_landmarks:
        position_mouse(results.multi_hand_landmarks)
        click(results.multi_hand_landmarks, mouse_event_threshold_angle)
        cv2.putText(
            image, text=f'Hand Detected, {is_detected}, {is_pressed}', org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(255, 255, 255), thickness=3
        )
        mp_drawing.draw_landmarks(
            image, results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )
        # print(detect_bend(results.multi_hand_landmarks, 0.01))

    cv2.imshow('IMAGE', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()


time_diff = dt.datetime.today().timestamp() - start_time
print(count / time_diff)
