import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 모델 로드
model = load_model('fall_detection_model.keras')

# 카메라 한 대 설정
cap = cv2.VideoCapture(1)

# 원본 영상 FPS, 해상도 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # 비디오 저장 객체 초기화

# 랜드마크 전처리 함수
def preprocess_landmarks(landmarks):
    flattened = np.array([lm.x for lm in landmarks] +
                         [lm.y for lm in landmarks] +
                         [lm.z for lm in landmarks])
    return flattened.reshape(1, -1)

fall_detected_start_time = None
fall_end_time = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Mediapipe Pose 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        input_data = preprocess_landmarks(landmarks)

        # 모델 예측
        prediction = model.predict(input_data, verbose=0)
        fall_probability = prediction[0][0]

        # 랜드마크 기반 Bounding Box 계산
        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        xmin = int(min(x_vals) * frame_width)
        xmax = int(max(x_vals) * frame_width)
        ymin = int(min(y_vals) * frame_height)
        ymax = int(max(y_vals) * frame_height)

        if fall_probability > 0.5:
            # 낙상 상태 처리
            if fall_detected_start_time is None:
                fall_detected_start_time = current_time
            fall_duration = current_time - fall_detected_start_time

            # 빨간색 컨투어박스 + "fall" 표시
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
            cv2.putText(frame, "fall", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            # 3초 이상 지속되면 Warning 표시 + 반짝임 효과
            if fall_duration >= 3:
                blink_intensity = int(255 * abs(np.sin(current_time * 5)))
                blink_color = (0, 0, blink_intensity)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), blink_color, 4)

                warning_text = "WARNING"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = xmin + (xmax - xmin - text_size[0]) // 2
                text_y = ymin + (ymax - ymin + text_size[1]) // 2
                cv2.putText(frame, warning_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, blink_color, 3, cv2.LINE_AA)

                # 이벤트 기반 녹화 시작
                if not recording:
                    print("녹화 시작")
                    out = cv2.VideoWriter(f'fall_event_{int(time.time())}.avi', fourcc, fps, (frame_width, frame_height))
                    recording = True

        else:
            # No Fall 상태 처리
            if fall_detected_start_time is not None:
                fall_end_time = current_time
            fall_detected_start_time = None

            # 흰색 컨투어박스 + "no_fall" 표시
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            cv2.putText(frame, "no_fall", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # 녹화 중이면 계속 기록 + 종료 타이밍 확인
        if recording:
            out.write(frame)
            if fall_end_time and current_time - fall_end_time >= 5:
                print("녹화 종료")
                out.release()
                out = None
                recording = False
                fall_end_time = None

    cv2.imshow('Camera - Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()

print("프로그램이 종료되었습니다.")
