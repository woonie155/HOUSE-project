# 평균 95.7%의 정밀도
import cv2
import mediapipe as mp #오픈cv
import numpy as np
import time, os

actions = ['ALT_F4', 'ALT_TAB', 'Full', 'Mute']
seq_length = 30 #윈도우 사이즈
secs_for_action = 60 #액션당 녹화시간 30초 #학습잘안되면 늘려도댐

#(455,100) = 모은 데이터개수, 판독위한 포인트개수들(손가락 각도15,랜드마크 데이터63,
# 랜드마크 visibility21(손가락 포인트점), 정답라벨1 모두 합치면 100개) , 랜드마크는 xyz로 표시댈듯+각도인식까지?.
#x와 y는 각각 영상 폭과 높이에 의해 [0.0, 1.0]으로 정규화됩니다.
#z는 랜드마크 깊이를 나타내며 손목의 깊이가 원점이며 값이 작을수록 랜드마크가 카메라에 더 가깝습니다. 
#z의 크기는 x와 거의 같은 척도를 사용합니다.
#학습(425,30,100) = 30은 LSTM에의해 시퀀스길이필요해 30사이즈 윈도우

# MediaPipe hands model  초기화
mp_drawing = mp.solutions.drawing_utils #랜드마크 그려주기
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    max_num_hands=1, #한손만
    min_detection_confidence=0.6, #손가락 감지 최소 신뢰값(0~1 높을수록정확한데, 감지확실해야해서 인식안될수도)
    min_tracking_confidence=0.6 #손가락 추적 최소 신뢰값
    )

#웹캡 초기화 설정
cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('./dataset', exist_ok=True)

while cap.isOpened(): #데이터 캠으로 촬영
    for idx, action in enumerate(actions):
        data = []
        ret, img = cap.read() #한프레임씩 읽기
        img = cv2.flip(img, 1) #좌우반전

        cv2.putText(img, f'{action.upper()}의 제스쳐를 취해주세요...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img) #모니터에 이미지 출력
        cv2.waitKey(1500) #1.5초 대기 (어떤액션취할지 준비)

        start_time = time.time()
        while time.time() - start_time < secs_for_action: # 반복시간설정
            ret, img = cap.read()
            img = cv2.flip(img, 1) #좌우 반전 (거울)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #오픈cvBGR->미디어파이프RGB
            result = hands.process(img) #이미지 전처리 #결과 수집반환.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #다시변경

            if result.multi_hand_landmarks is not None: #인식되면
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] #visibility는 각도인식됐는지판별

                    # Compute angles between joints #손 각도 계산 알고리즘
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint - 각도인식빼고
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product 
                    angle = np.arccos(np.einsum('nt,nt->n', #행 합 연산
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    angle = np.degrees(angle) # Convert radian to degree
##########################################################
                    #라벨생성
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) #라벨생성 idx=0,1,2

                    d = np.concatenate([joint.flatten(), angle_label]) #100x100행렬로
                    data.append(d)
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data) #np로변환 반복마다 모은 제스쳐동작 저장
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data) #npy로 저장

        # Create sequence data #시퀀스데이터이용해 모델에 학습시킬것.
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data) 
    break
