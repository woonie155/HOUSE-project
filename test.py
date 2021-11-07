import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui
import time, os
import math

actions = ['ALT_F4', 'ALT_TAB', 'ENTER']
seq_length = 30

model = load_model('models/model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
after_x = 0
after_y = 0
tmp=0
flag=0
tmp2=1


p_x_0 = 0; p_y_0 = 0; p_z_0 = 0 
p_x_8 = 0; p_y_8 = 0; p_z_8 = 0
point = 0; 
while cap.isOpened():
    if point == 1:
        break
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.putText(img, f' Measure the Point...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)
    cv2.waitKey(200)
    
    start_time = time.time()
    while time.time() - start_time < 3: #3초간 기준점 측정
        ret, img = cap.read()
        
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, j]           
                p_x_0 = joint[0, 0]; p_y_0 = joint[0, 1]; p_z_0 = joint[0, 2]
                p_x_8 = joint[8, 0]; p_y_8 = joint[8, 1]; p_z_8 = joint[8, 2]
            point = 1
                

# 2차평면에서의 각도를 구해야함.  4번과 0번의 각도 계산
p_x_0 *= 100; p_x_8 *= 100 # 각 점의 x좌표
p_y_0 *= 100; p_y_8 *= 100 # 각 점의 y좌표
p_y_0 = 100 - p_y_0; p_y_8 = 100-p_y_8
height = p_y_8 - p_y_0; dis = p_x_8 - p_x_0; 
stand_degree = math.atan2(height, dis)
stand_degree = math.degrees(stand_degree)


while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, j]            
                
            if tmp == 0:
                before_x = joint[8, 0]*100
                before_y = joint[8, 1]*100
                tmp=1
                p_x_0 = joint[0, 0]; p_y_0 = joint[0, 1]; p_z_0 = joint[0, 2]
                p_x_8 = joint[8, 0]; p_y_8 = joint[8, 1]; p_z_8 = joint[8, 2]
                p_x_0 *= 100; p_x_8 *= 100 # 각 점의 x좌표
                p_y_0 *= 100; p_y_8 *= 100 # 각 점의 y좌표
                p_y_0 = 100 - p_y_0; p_y_8 = 100 - p_y_8
                height = p_y_8 - p_y_0; dis = p_x_8 - p_x_0
                be_cur = math.atan2(height, dis) 
                be_cur = math.degrees(be_cur)
            else:
                after_x = before_x - joint[8, 0]*100
                after_y = before_y - joint[8, 1]*100
                tmp=0
                flag = 1
                p_x_0 = joint[0, 0]; p_y_0 = joint[0, 1]; p_z_0 = joint[0, 2]
                p_x_8 = joint[8, 0]; p_y_8 = joint[8, 1]; p_z_8 = joint[8, 2]
                p_x_0 *= 100; p_x_8 *= 100 # 각 점의 x좌표
                p_y_0 *= 100; p_y_8 *= 100 # 각 점의 y좌표
                p_y_0 = 100 - p_y_0; p_y_8 = 100 - p_y_8
                height = p_y_8 - p_y_0; dis = p_x_8 - p_x_0
                cur = math.atan2(height, dis) 
                cur = math.degrees(cur)


            if  flag == 1:
                com = (be_cur-cur)
                plus = stand_degree - cur
                if (abs(after_x)<8) and (abs(after_y)<8):
                    pyautogui.move(-after_x*30, -after_y*30 )
                    if(abs(after_x*30)<20 and abs(after_y*30)<20):
                        if cur>90: 
                            pyautogui.move(-20,0)
                        else:
                            pyautogui.move(20,0)

                    print(-after_x, -after_y)
                    flag=0







            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]


            #pyautogui.move(v[4,0] ,  )

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n', #행별로 벡터연산
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            #30사이즈 다모이면.        
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9: #90넘어야지만 저장
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?' #판단안되는경우
            if action_seq[-1] == action_seq[-2] == action_seq[-3]: #마지막세개가 똑같은경우만 정확한제스쳐라판단
                this_action = action
            else:
                pass

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break