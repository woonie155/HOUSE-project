import cv2, sys
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui as pg
import math
import datetime as dt
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from win32api import GetSystemMetrics
import mouse
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QMovie
import threading

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_ex = volume.GetVolumeRange()
volume_range = [volume_ex[0], (volume_ex[0]+volume_ex[1])/2, volume_ex[1]]


gesture = ['ALT_TAB', 'ALT_F4', 'FULL', 'SOUND_CONTROL']
seq_length = 30
model = load_model('models/cursor_model_t1.h5')
seq = []
gesture_seq = []
light_flag = False
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
mouse_event_threshold_angle = [0.5, 0.7, 0.7, 0.3, 0.7]
duration_threshold = 5
duration_threshold_ring = 20
duration = [0, 0, 0, 0, 0]

mouse_pressed = [False, mouse.is_pressed('left'), mouse.is_pressed('middle'), mouse.is_pressed('right'), False]
finger_bend = [False, False, False, False, False]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)


def euc_dist_2d(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)


def arithmetic_mean(landmarks):
    return sum(landmark.x for landmark in landmarks) / len(landmarks),\
        sum(landmark.y for landmark in landmarks) / len(landmarks)


def position_mouse(landmarks):
    cursor_position_list_x.pop(0)
    cursor_position_list_y.pop(0)
    cursor_position_x, cursor_position_y = arithmetic_mean(list((landmarks.landmark[0], landmarks.landmark[17])))
    cursor_position_list_x.append((((cursor_position_x - 0.5) / ratio) + 0.5) * display_width)
    cursor_position_list_y.append((((cursor_position_y - 0.5) / ratio) + 0.5) * display_height)
    mouse.move(sum(cursor_position_list_x) / len(cursor_position_list_x), sum(cursor_position_list_y) / len(cursor_position_list_y))


def is_bend(hand, threshold):
    if (hand.landmark[2].x > hand.landmark[0].x) and (hand.landmark[2].x > hand.landmark[4].x):
        finger_bend[0] = True
    elif (hand.landmark[2].x < hand.landmark[0].x) and (hand.landmark[2].x < hand.landmark[4].x):
        finger_bend[0] = True
    elif hand.landmark[4].x > hand.landmark[6].x:
        finger_bend[0] = True
    else:
        finger_bend[0] = False

    v_out = np.array([
        hand.landmark[5].x - hand.landmark[8].x,
        hand.landmark[5].y - hand.landmark[8].y,
        hand.landmark[5].z - hand.landmark[8].z
    ])
    v_in = np.array([
        hand.landmark[0].x - hand.landmark[5].x,
        hand.landmark[0].y - hand.landmark[5].y,
        hand.landmark[0].z - hand.landmark[5].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold[1]:
        finger_bend[1] = True
    else:
        finger_bend[1] = False

    v_out = np.array([
        hand.landmark[9].x - hand.landmark[12].x,
        hand.landmark[9].y - hand.landmark[12].y,
        hand.landmark[9].z - hand.landmark[12].z
    ])
    v_in = np.array([
        hand.landmark[0].x - hand.landmark[9].x,
        hand.landmark[0].y - hand.landmark[9].y,
        hand.landmark[0].z - hand.landmark[9].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold[2]:
        finger_bend[2] = True
    else:
        finger_bend[2] = False


    v_out = np.array([
        hand.landmark[13].x - hand.landmark[16].x,
        hand.landmark[13].y - hand.landmark[16].y,
        hand.landmark[13].z - hand.landmark[16].z
    ])
    v_in = np.array([
        hand.landmark[0].x - hand.landmark[13].x,
        hand.landmark[0].y - hand.landmark[13].y,
        hand.landmark[0].z - hand.landmark[13].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold[3]:
        finger_bend[3] = True
    else:
        finger_bend[3] = False

    v_out = np.array([
        hand.landmark[17].x - hand.landmark[20].x,
        hand.landmark[17].y - hand.landmark[20].y,
        hand.landmark[17].z - hand.landmark[20].z
    ])
    v_in = np.array([
        hand.landmark[0].x - hand.landmark[17].x,
        hand.landmark[0].y - hand.landmark[17].y,
        hand.landmark[0].z - hand.landmark[17].z
    ])
    v_out = v_out / np.linalg.norm(v_out)
    v_in = v_in / np.linalg.norm(v_in)
    if np.dot(v_out, v_in) < threshold[4]:
        finger_bend[4] = True
    else:
        finger_bend[4] = False


def click():
    global duration
    if finger_bend[1]:
        if not mouse_pressed[1]:
            if duration[1] < duration_threshold:
                duration[1] += 1
            else:
                mouse.press('left')
                mouse_pressed[1] = True
                duration[1] = 0
    else:
        if mouse_pressed[1]:
            mouse.release('left')
            mouse_pressed[1] = False
        elif duration[1] != 0:
            mouse.click('left')
            duration[1] = 0

    if finger_bend[2]:
        if not mouse_pressed[2]:
            if duration[2] < duration_threshold:
                duration[2] += 1
            else:
                mouse.press('middle')
                mouse_pressed[2] = True
                duration[2] = 0
    else:
        if mouse_pressed[2]:
            mouse.release('middle')
            mouse_pressed[2] = False
        elif duration[2] != 0:
            mouse.click('middle')
            duration[2] = 0

    if finger_bend[3]:
        if not mouse_pressed[3]:
            if duration[3] < duration_threshold_ring:
                duration[3] += 1
            else:
                mouse.click('right')
                mouse_pressed[3] = True
                duration[3] = 0
    else:
        if mouse_pressed[3]:
            mouse_pressed[3] = False


def arithMean(landmarks):
    sum_x = 0
    sum_y = 0
    for landmark in landmarks:
        sum_x += landmark.x
        sum_y += landmark.y
    return sum_x / 21, sum_y / 21


#키보트 이벤트
keyboard_event_threshold = 0.08
key_flag = 0
tab_flag = 0


def keyboard_event_flag(landmarks):
    if finger_bend[0] and finger_bend[3] and finger_bend[4]:
        return 1
    if (abs(landmarks[4].x - landmarks[16].x)<=keyboard_event_threshold and abs(landmarks[4].x - landmarks[20].x)<=keyboard_event_threshold
            and abs(landmarks[20].x - landmarks[16].x)<=keyboard_event_threshold):
            if (abs(landmarks[4].y - landmarks[16].y)<=keyboard_event_threshold and abs(landmarks[4].y - landmarks[20].y)<=keyboard_event_threshold
            and abs(landmarks[20].y - landmarks[16].y)<=keyboard_event_threshold) and (landmarks[20].y > landmarks[18].y):
                return 1
    if (landmarks[2].x < landmarks[4].x):
        return 1
    if (landmarks[6].x < landmarks[4].x):
        return 1
    if finger_bend[0] or (landmarks[20].y > landmarks[18].y):
        return 1
    return 0


def volumn_value(landmarks):
    x_5 = landmarks[5].x * 100
    y_5 = landmarks[5].y * 100
    x_8 = landmarks[8].x * 100
    y_8 = landmarks[8].y * 100
    tanp = y_5 - y_8
    tanc = x_8 - x_5
    degree = int(math.degrees(math.atan(tanc/tanp)))
    v = volume_range[0] + (degree + 23) * (abs(volume_range[2] - volume_range[0]) / 30)
    if v >= volume_range[2]:
        v = volume_range[2]
    if v <= volume_range[0]:
        v = volume_range[0]
    return v


def alt_tab(landmarks):
    global tab_flag
    if (landmarks[8].y < landmarks[14].y):
        tab_flag = 0
    if (tab_flag==0)and(landmarks[8].y > landmarks[14].y):
        pg.press('tab')
        tab_flag = 1

class light_green(QtWidgets.QMainWindow):
    global light_flag
    def __init__(self, img1, xy, size, on_top): 
        super(light_green, self).__init__()
        self.light_flag = 0
        self.timer = QtCore.QTimer(self)
        self.img1 = img1
        self.xy = xy
        self.direction = [0, 0]
        self.size = size
        self.on_top = on_top
        self.localPos = None
        self.setupUi()
        self.show()
    
    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        
        f = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint 
                                    if self.on_top else QtCore.Qt.FramelessWindowHint)       
        self.setWindowFlags(f)
        
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        
        label = QtWidgets.QLabel(centralWidget)
        show = QMovie(self.img1)
        label.setMovie(show)
        show.setScaledSize(QtCore.QSize(int(100*self.size), int(100*self.size)))
        show.start()
        self.setGeometry(self.xy[0], self.xy[1], int(100*self.size),int(100*self.size))
        
    def convert_l(self):
        global light_flag
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.__Handler)
        self.timer.start(int(10))
    def __Handler(self): 
        global light_flagf
        if light_flag == False:
            self.xy[1] = -50           
        else:
            self.xy[1] = 0           
        self.move(*self.xy)
    def mouseDoubleClickEvent(self, event):
        sys.exit()
        
global flag_exit
flag_exit = False
def th_1():
    global light_flag
    global flag_exit
    app = QtWidgets.QApplication(sys.argv)
    green = light_green('green1.gif', xy=[display_width-50, -50], on_top=True, size=0.3)
    green.convert_l()
    exe_p = light_green('exit.gif', xy=[display_width-100, display_height-100], on_top=False, size=1)
    while True:
        if flag_exit: 
            app.quit()
            break
        sys.exit(app.exec_())
t1 = threading.Thread(target=th_1)
t1.daemon=True
t1.start()


while cap.isOpened():
    light_flag = False
    ret, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks is not None:
        light_flag = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            is_bend(hand_landmarks, mouse_event_threshold_angle)
            if not keyboard_event_flag(hand_landmarks.landmark):
                seq = []
                if key_flag == 1 or key_flag == 2:
                    key_flag = 0
                    pg.keyUp('alt')
                position_mouse(hand_landmarks)
                click()
                cv2.putText(
                    image, text='Mouse mode', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(255, 255, 255), thickness=2)
            else:
                if key_flag == 2:
                    alt_tab(hand_landmarks.landmark)
                else:
                    joint = np.zeros((21, 2))
                    for j, landmark in enumerate(hand_landmarks.landmark):
                        joint[j] = [landmark.x, landmark.y]
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :2]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :2]
                    v = v2 - v1 # [20, 2]
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
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
                            this_gesture = key_event
                            if this_gesture == gesture[0]:
                                pg.keyDown('alt')
                                pg.press('tab')
                                key_flag = 2
                            elif this_gesture == gesture[1]:
                                pg.hotkey('alt', 'F4')
                                key_flag = 1
                            elif this_gesture == gesture[2]:
                                pg.press('f')
                                key_flag = 1
                            else:
                                v_value = volumn_value(hand_landmarks.landmark)
                                volume.SetMasterVolumeLevel(v_value, None)
                            cv2.putText(
                                image, f'{this_gesture.upper()}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', image)
    if cv2.waitKey(1) == ord('q'):
        flag_exit = True
        cap.release()
        break
sys.exit()
