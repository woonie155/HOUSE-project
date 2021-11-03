import cv2
import mediapipe as mp
import math
import pyautogui as pg

pg.FAILSAFE = False

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camWidth, camHeight = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winWidth, winHeight = pg.size()
curX, curY = pg.position()
ratio = 0.5

def eucDist2d(x0, x1, y0, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def arithMean(landmarks):
    sum_x = 0
    sum_y = 0
    for landmark in landmarks:
        sum_x += landmark.x
        sum_y += landmark.y
    return sum_x / 21, sum_y / 21

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), 1)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                cv2.putText(
                    image, text='Hand Detected',
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 0, 0), thickness=2
                )
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                curX, curY = arithMean(hand_landmarks.landmark)
                curX = (((curX - 0.5) / ratio) + 0.5) * winWidth
                curY = (((curY - 0.5) / ratio) + 0.5) * winHeight
                pg.moveTo(int(curX), int(curY))

                if eucDist2d(hand_landmarks.landmark[8].x,
                            hand_landmarks.landmark[4].x,
                            hand_landmarks.landmark[8].y,
                            hand_landmarks.landmark[4].y) < 0.01:
                    pg.click()


        cv2.imshow('IMAGE', image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()