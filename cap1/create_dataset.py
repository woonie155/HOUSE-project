# 평균 95.7%의 정밀도
import cv2
import mediapipe as mp 
import numpy as np
import time, os

gesture = ['ALT_TAB',  'ALT_F4',  'FULL',  'SOUND_CONTROL']
seq_length = 30 
secs_for_gesture = 30

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('./dataset', exist_ok=True)

while cap.isOpened(): 
    for idx, action in enumerate(gesture):
        data = []
        ret, img = cap.read() 
        img = cv2.flip(img, 1)

        cv2.putText(img, f'Make a {action.upper()} gesture...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img) 
        cv2.waitKey(1500)

        start_time = time.time()
        while time.time() - start_time < secs_for_gesture: 
            ret, img = cap.read()
            img = cv2.flip(img, 1) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None: 
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 2))
                    for i, l_m in enumerate(res.landmark):
                        joint[i] = [l_m.x, l_m.y] 
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] 
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] 
                    v = v2 - v1 
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 
                    angle = np.degrees(angle) 

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)
                    add = np.concatenate([joint.flatten(), angle_label])
                    data.append(add)
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break   
        
        data = np.array(data) 
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data) 
    break
