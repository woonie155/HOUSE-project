import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

actions = ['ALT_F4', 'ALT_TAB', 'Full', 'Mute']
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

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []
after_x = 0
after_y = 0
tmp=0
flag=0

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
                before_x = joint[4,0]*100
                before_y = joint[4,1]*100
                tmp=1
            else:
                after_x = before_x - joint[4, 0]*100
                after_y = before_y - joint[4, 1]*100
                tmp=0
                flag = 1
            
            if  flag == 1:
                if(abs(after_x)>8 ) or (abs(after_y)>8):
                    continue
                pyautogui.move(-after_x*30, -after_y*30)
                print(after_x, after_y)
                flag=0
                


            엄검x = abs(joint[4,0]*100-joint[8,0]*100)
            엄검y = abs(joint[4,1]*100-joint[8,1]*100)
            엄검z = abs(joint[4,2]*100-joint[8,2]*100)

            엄중x = abs(joint[4,0]*100-joint[12,0]*100)
            엄중y = abs(joint[4,1]*100-joint[12,1]*100)
            엄중z = abs(joint[4,2]*100-joint[12,2]*100)

            엄약x = abs(joint[4,0]*100-joint[16,0]*100)
            엄약y = abs(joint[4,1]*100-joint[16,1]*100)
            엄약z = abs(joint[4,2]*100-joint[16,2]*100)

            # #좌클릭
            # if 엄검x < 2 and 엄검y <2 and 엄검z <4.5:
            #     pyautogui.click()

            # #우클릭
            # if 엄중x < 2 and 엄중y <2 and 엄중z <4.5:
            #     pyautogui.rightClick()

            # #휠클릭
            # if 엄약x < 2 and 엄약y <2 and 엄약z <4.5:
            #     pyautogui.middleClick()

            #커서이동
            #pyautogui.move((joint[4,0]*10-5)*2, (joint[4,1]*10-5)*2)



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
                # pyautogui.move((joint[4,0]*10-5)*10, (joint[4,1]*10-5)*10)
                #모니터pyautogui()
            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break