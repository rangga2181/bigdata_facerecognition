# 😊 Quality-Aware Real-Time FER
## Facial Expression Recognition + Confidence Calibration + Reject Option

> EfficientNet-B0 · Temperature Scaling · Triple-Threshold Reject Option · MediaPipe · Streamlit

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset
Letakkan dataset dalam format:
```
datasets/
  fer2013/
    train/ val/ test/
      Angry/ Disgust/ Fear/ Happy/ Sad/ Surprise/ Neutral/
```
Atau gunakan file `fer2013.csv` yang bisa diunduh dari Kaggle.

### 3. Training
```bash
python train.py --config configs/fer2013.yaml
# Jika menggunakan CSV:
python train.py --config configs/fer2013.yaml --csv datasets/fer2013/fer2013.csv
```

### 4. Calibration
```bash
python calibrate.py --checkpoint checkpoints/best_model.pth
```

### 5. Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --temperature checkpoints/temperature.pth
```

### 6. Demo Webcam
```bash
python demo.py --checkpoint checkpoints/best_model.pth \
               --temperature checkpoints/temperature.pth
```

### 7. Streamlit App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
fer-project/
├── datasets/              ← FER2013 / RAF-DB dataset
├── checkpoints/           ← Saved model weights
│   └── logs/              ← Training logs
├── notebooks/             ← Jupyter notebooks (EDA, training)
├── configs/               ← YAML configuration files
│   ├── default.yaml
│   ├── fer2013.yaml
│   └── rafdb.yaml
├── src/
│   ├── detection/         ← Face detection (MediaPipe, RetinaFace)
│   ├── alignment/         ← Face alignment (affine transform)
│   ├── quality/           ← Quality assessment (blur, brightness, pose, occlusion)
│   ├── fer/               ← FER model, dataset, training
│   ├── calibration/       ← Temperature scaling, calibration evaluator
│   ├── rejection/         ← Reject option (triple threshold)
│   ├── realtime/          ← Webcam pipeline, visualizer
│   └── utils/             ← Metrics, logger, config loader
├── results/               ← Evaluation plots & reports
├── app.py                 ← Streamlit web app
├── train.py               ← Training entry point
├── calibrate.py           ← Calibration entry point
├── evaluate.py            ← Evaluation entry point
├── demo.py                ← Webcam demo entry point
└── requirements.txt
```

---

## 🧩 System Architecture

```
Webcam Stream
      ↓
Face Detection (MediaPipe / RetinaFace)
      ↓
Face Alignment (eye landmark + affine)
      ↓
Quality Assessment Module
  ├── Blur Score (Laplacian variance)
  ├── Brightness Score (YCrCb luminance)
  ├── Pose Score (solvePnP euler angles)
  ├── Occlusion Score (landmark visibility)
  └── Face Size Score
      ↓
Reject Low Quality Frame (quality_score < q_threshold)
      ↓
FER Backbone (EfficientNet-B0)
      ↓
Confidence Calibration (Temperature Scaling, T optimal)
      ↓
Reject Option
  ├── Reject if confidence < c_threshold
  └── Reject if entropy > e_threshold
      ↓
Final Emotion Prediction
```

---

## 📊 Emotion Classes

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

---

## ⌨️ Webcam Shortcuts

| Key | Action |
|-----|--------|
| `q` / `ESC` | Keluar |
| `s` | Screenshot |
| `p` | Toggle probability panel |

---

## 📝 Author
[Your Name] — [Your University] — 2026
