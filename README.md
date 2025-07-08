# UWC 2025 AI & 무인이동체 퓨처 해커톤 소스 코드_인텔이 벌벌 구글이 전전긍긍

## 📋 프로젝트 개요

**드론 & 딥러닝 기반 통합 응급상황 대응 시스템**은 축제, 공사현장, 해변 등 기존 모니터링 시설 구축이 어려운 공간에서 드론과 AI 기술을 활용해 응급사고를 실시간 감지하고 구조를 지원하는 시스템입니다.

## 🎯 주요 기능

- **익수 상황 감지**: 딥러닝 모델을 통한 실시간 익수 상황 탐지
- **실신 사고 감지**: 딥러닝 모델을 통한 실시간 실신 상황 탐지

## 🛠️ 기술 스택

- **딥러닝**: Python, TensorFlow + Keras, PyTorch, YOLOv5
- **컴퓨터 비전**: OpenCV, Mediapipe
- **개발 환경**: Google Colab, Spyder

## 📁 프로젝트 구조

```
├── best.pt                           # 최적화된 익수 감지 훈련 모델
├── drowning_model_predict.ipynb      # 익수 상황 예측 모델
├── drowning_model_train.ipynb        # 익수 상황 학습 모델
├── fall_detection_model.keras        # 최적화된 실신 감지 훈련 모델
├── falling_detection_test.py         # 실신 감지 테스트 스크립트
└── falling_detection_train.py        # 실신 감지 학습 스크립트
```

## 🔧 설치 및 필요 환경

### 필수 요구사항
```bash
Python 3.8+
```

### 필요 라이브러리 설치
```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn matplotlib
```

## 🚀 사용 방법

### 1. 익수 상황 감지 (Google Colab)

#### 모델 학습
1. [Google Colab](https://colab.research.google.com/) 접속
2. `drowning_model_train.ipynb` 파일 업로드
3. 런타임 → 런타임 유형 변경 → GPU 선택 (권장)
4. 모든 셀 실행

#### 실시간 예측
1. [Google Colab](https://colab.research.google.com/) 접속
2. `drowning_model_predict.ipynb` 파일 업로드
3. 학습된 모델(`best.pt`) 파일을 Colab에 업로드
4. 예측 셀 실행

### 2. 실신 사고 감지 (로컬 환경)

#### 모델 학습
```python
# 모델 학습 실행
python falling_detection_train.py
```

#### 모델 테스트
```python
# 테스트 실행
python falling_detection_test.py
```

## 🎯 모델 성능

### 익수 감지 모델 성능

#### Confusion Matrix
|              | Drowning | Swimming | Not in Water | Background |
|--------------|----------|----------|--------------|------------|
| **Drowning** | **0.91** | 0.03     | 0.03         | 0.04       |
| **Swimming** | 0.03     | **0.94** | 0.02         | 0.01       |
| **Not in Water** | 0.01 | 0.01     | **0.96**     | 0.02       |
| **Background** | 0.05   | 0.03     | 0.13         ||

**주요 성능 지표:**
- 익수 감지 정확도: **91%**
- 수영 감지 정확도: **94%**
- 물밖 상황 감지 정확도: **96%**

### 실신 감지 모델 성능

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **fall** | 0.96 | 0.98 | 0.97 | 2380 |
| **not_fall** | 0.96 | - | 0.95 | 1489 |
| **accuracy** | - | - | **0.96** | 3869 |
| **macro avg** | 0.96 | 0.96 | 0.96 | 3869 |
| **weighted avg** | 0.96 | 0.96 | 0.96 | 3869 |

**주요 성능 지표:**
- 전체 정확도: **96%**
- 실신 감지 재현율: **98%** (중요한 상황 놓치지 않음)
- 두 클래스 모두 균형잡힌 정밀도: **96%**

## 📊 시스템 특징

- **실시간 처리**: 저지연 실시간 영상 분석
- **높은 정확도**: 딥러닝 기반 정밀한 상황 인식

## 🎥 데모

### 익수 감지 시연
- Google Colab을 활용한 실시간 시연
- 4가지 상황 분류: 익수, 수영, 물밖

### 실신 감지 시연
- Mediapipe를 활용한 실시간 자세 분석
- 실신 상황 즉시 알림
