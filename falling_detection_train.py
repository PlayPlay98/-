#실신감지 최종 모델학습 코드
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.callbacks import ReduceLROnPlateau




# Mediapipe 솔루션 초기화 해주기
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 이미지를 RGB로 변환해주기
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = [lm.x for lm in results.pose_landmarks.landmark] + \
                        [lm.y for lm in results.pose_landmarks.landmark] + \
                        [lm.z for lm in results.pose_landmarks.landmark]
            landmarks_list.append(landmarks)
    
    cap.release()
    return landmarks_list

def load_data(data_dir, label_map):
    X, y = [], []
    for label, class_dir in label_map.items():
        class_path = os.path.join(data_dir, class_dir)
        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)
            landmarks = extract_landmarks(video_path)
            if landmarks:
                X.extend(landmarks)
                y.extend([label] * len(landmarks))
    
    return np.array(X), np.array(y)

# 데이터셋 경로와 레이블 맵
data_dir = 'C:/Users/82104/Desktop/baekseok/2024 4-1/딥러닝 프로그래밍/dataset'
label_map = {0: 'fall', 1: 'not_fall'}

# 데이터 로드해주기
X, y = load_data(data_dir, label_map)

# 데이터 정규화해주기
X = np.array(X)
y = np.array(y)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Flatten(input_shape=(len(X_train[0]),)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# 모델 훈련하기
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr])


# 모델 저장해주기
model.save('fall_detection_model.keras')

# 저장한 모델 불러오기
model = tf.keras.models.load_model('fall_detection_model.keras')

# 테스트셋으로 평가하기
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# 테스트셋 예측
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix 생성
cm = confusion_matrix(y_test, y_pred_classes)
labels = ['fall', 'not_fall']

# Confusion Matrix 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 상세 분류 리포트 출력
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=labels))
