# Quality-Aware Real-Time Facial Expression Recognition
## with Confidence Calibration and Reject Option

---

# 1. Project Overview

## Background
Facial Expression Recognition (FER) merupakan salah satu bidang penting dalam computer vision yang digunakan untuk mengenali ekspresi wajah manusia secara otomatis. Namun, implementasi FER pada kondisi real-time webcam masih menghadapi berbagai tantangan seperti blur, pencahayaan buruk, occlusion, face misalignment, dan prediksi model yang terlalu percaya diri terhadap input yang tidak valid.

Project ini bertujuan membangun framework FER real-time yang lebih trustworthy dengan mengintegrasikan:

- Face Quality Assessment
- Confidence Calibration
- Reject Option Mechanism

Framework diharapkan mampu memberikan prediksi ekspresi yang lebih andal pada kondisi webcam in-the-wild.

---

# 2. Objectives

## Main Objective
Membangun sistem real-time Facial Expression Recognition berbasis deep learning yang mampu:

- mengenali ekspresi wajah,
- mengevaluasi kualitas frame,
- mengkalibrasi confidence prediction,
- menolak prediksi yang tidak reliable.

---

# 3. Research Contributions

## Proposed Contributions

### 1. Quality-Aware FER
Mengintegrasikan quality assessment sebelum proses klasifikasi ekspresi.

### 2. Confidence Calibration
Menggunakan temperature scaling untuk meningkatkan reliability confidence score.

### 3. Reject Option
Menerapkan dual-threshold rejection berdasarkan:
- kualitas frame,
- uncertainty prediction.

### 4. Real-Time FER Framework
Mengembangkan pipeline FER realtime berbasis webcam.

---

# 4. Proposed System Architecture

```text
Webcam Stream
      ↓
Face Detection
      ↓
Face Alignment
      ↓
Quality Assessment Module
      ↓
Reject Low Quality Frame
      ↓
FER Backbone Model
      ↓
Confidence Calibration
      ↓
Reject Option
      ↓
Final Emotion Prediction
```

---

# 5. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Face Detection | MediaPipe / RetinaFace |
| Model Backbone | EfficientNet-B0 |
| Deployment | Streamlit / Flask |
| Training Environment | Google Colab / Local GPU |

---

# 6. Dataset

## Main Dataset

### FER2013
- 7 emotion classes
- grayscale images
- lightweight dataset

### RAF-DB
- real-world facial expressions
- more robust for in-the-wild scenarios

---

# 7. Emotion Classes

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

# 8. Project Modules

## 1. Face Detection Module
Mendeteksi wajah dari webcam stream.

### Candidate Methods
- MediaPipe Face Detection
- RetinaFace

---

## 2. Face Alignment Module
Melakukan alignment agar posisi wajah lebih konsisten.

### Techniques
- Eye landmark alignment
- Affine transformation

---

## 3. Quality Assessment Module

### Parameters
- Blur detection
- Brightness score
- Face size
- Occlusion estimation
- Pose estimation

### Output
Quality score:
```text
0.0 → poor quality
1.0 → high quality
```

---

## 4. FER Backbone Module

### Candidate Models
- MobileNetV3
- EfficientNet-B0
- ConvNeXt Tiny

### Selected Model
EfficientNet-B0 karena:
- ringan,
- cepat,
- cocok untuk realtime inference.

---

## 5. Confidence Calibration Module

### Method
Temperature Scaling

### Purpose
Mengurangi overconfidence prediction.

---

## 6. Reject Option Module

### Rejection Criteria

#### Quality Threshold
```text
Reject if quality_score < q_threshold
```

#### Confidence Threshold
```text
Reject if confidence < c_threshold
```

#### Entropy Threshold
```text
Reject if entropy > e_threshold
```

---

# 9. Training Pipeline

```text
Dataset
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
Model Training
   ↓
Validation
   ↓
Calibration
   ↓
Evaluation
```

---

# 10. Preprocessing

## Techniques
- Face cropping
- Resize (224x224)
- Normalization
- Histogram equalization

---

# 11. Data Augmentation

## Augmentations
- Horizontal flip
- Rotation
- Brightness adjustment
- Gaussian blur
- Random occlusion

Purpose:
mensimulasikan kondisi webcam nyata.

---

# 12. Evaluation Metrics

## Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- UAR (Unweighted Average Recall)

## Calibration Metrics
- Expected Calibration Error (ECE)
- Brier Score

## Selective Prediction Metrics
- Coverage
- Selective Risk

## Real-Time Metrics
- FPS
- Latency

---

# 13. Folder Structure

```text
project/
│
├── datasets/
├── models/
├── checkpoints/
├── notebooks/
├── src/
│   ├── detection/
│   ├── alignment/
│   ├── quality/
│   ├── fer/
│   ├── calibration/
│   ├── rejection/
│   └── realtime/
│
├── configs/
├── results/
├── app.py
├── train.py
├── evaluate.py
└── requirements.txt
```

---

# 14. Project Timeline

| Week | Task |
|---|---|
| 1 | Literature review |
| 2 | Dataset preparation |
| 3 | Preprocessing pipeline |
| 4 | Baseline FER training |
| 5 | Quality assessment module |
| 6 | Confidence calibration |
| 7 | Reject option integration |
| 8 | Real-time webcam integration |
| 9 | Evaluation |
| 10 | Documentation |

---

# 15. Expected Results

## Expected Outcomes

- FER model with stable real-time inference
- Better confidence reliability
- Reduced false confident predictions
- Selective prediction capability
- Improved robustness under webcam conditions

---

# 16. Future Work

Possible future improvements:
- Temporal FER using video transformers
- Multimodal emotion recognition
- Speech + facial emotion fusion
- Edge deployment optimization
- Mobile deployment

---

# 17. References

## Suggested Papers

1. Facial Expression Recognition using CNN
2. Temperature Scaling for Confidence Calibration
3. Selective Classification for Deep Neural Networks
4. Face Image Quality Assessment
5. Real-Time FER in the Wild

---

# 18. Author

## Researcher
[Your Name]

## Institution
[Your University]

## Year
2026
