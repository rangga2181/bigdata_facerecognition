"""
app.py
Streamlit Web App untuk FER — upload image & live webcam mode.

Usage:
    streamlit run app.py
"""
import sys
import io
import time
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

import streamlit as st

sys.path.insert(0, ".")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quality-Aware FER",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 6px 0;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #7EB8F7; }
.metric-label { font-size: 0.8rem; color: #aaa; margin-top: 4px; }

.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 8px auto;
}

.status-accepted { background: #0e4f2d; color: #4ade80; border: 1px solid #4ade80; }
.status-rejected  { background: #4f1313; color: #f87171; border: 1px solid #f87171; }

.quality-bar-bg {
    background: #2a2a3e;
    border-radius: 8px;
    height: 10px;
    margin: 4px 0;
    overflow: hidden;
}
.quality-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

EMOTION_EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😄",
    "Sad":      "😢",
    "Surprise": "😲",
    "Neutral":  "😐",
}

EMOTION_COLORS = {
    "Angry":    "#ef4444",
    "Disgust":  "#a855f7",
    "Fear":     "#f97316",
    "Happy":    "#22c55e",
    "Sad":      "#3b82f6",
    "Surprise": "#eab308",
    "Neutral":  "#94a3b8",
}


@st.cache_resource
def load_pipeline(model_path: str, temperature_path: str = None):
    """Load model dan semua komponen (di-cache agar tidak reload)."""
    from src.fer.model import FERModel
    from src.fer.transforms import get_inference_transform
    from src.calibration.temperature_scaling import TemperatureScaling
    from src.rejection.reject_option import RejectOption
    from src.quality.quality_aggregator import QualityAggregator
    from src.detection import get_detector
    from src.alignment.face_aligner import FaceAligner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FERModel.load_from_checkpoint(model_path, device=device)
    model.eval()

    calibrator = None
    if temperature_path and Path(temperature_path).exists():
        calibrator = TemperatureScaling.load(temperature_path)

    return {
        "model":      model,
        "calibrator": calibrator,
        "detector":   get_detector("mediapipe"),
        "aligner":    FaceAligner(),
        "quality":    QualityAggregator(),
        "rejector":   RejectOption(),
        "transform":  get_inference_transform(),
        "device":     device,
    }


def run_inference(pipeline, image_bgr: np.ndarray):
    """Jalankan inference pada satu gambar."""
    from src.rejection.reject_option import PredictionResult

    model      = pipeline["model"]
    calibrator = pipeline["calibrator"]
    detector   = pipeline["detector"]
    aligner    = pipeline["aligner"]
    quality    = pipeline["quality"]
    rejector   = pipeline["rejector"]
    transform  = pipeline["transform"]
    device     = pipeline["device"]

    detection = detector.detect_largest(image_bgr)

    if detection is None:
        return None, None, None

    bbox = detection.bbox
    face = aligner.align_from_bbox(image_bgr, bbox)
    if face is None:
        face = aligner.simple_crop(image_bgr, bbox)
    if face is None:
        return None, None, None

    q_report = quality.assess(face, bbox=bbox, frame_shape=image_bgr.shape[:2])

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor   = transform(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if calibrator:
            probs = calibrator.predict_proba(logits).cpu().numpy()[0]
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    from src.fer.dataset import EMOTION_CLASSES
    result = rejector.decide(probs, q_report.quality_score)
    return result, q_report, bbox


def draw_bbox_on_image(image_bgr, bbox, result):
    """Gambar bounding box di image."""
    if bbox is None:
        return image_bgr
    x, y, w, h = bbox
    color = (0, 220, 100) if (result and result.is_accepted) else (0, 80, 220)
    out = image_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    return out


def render_quality_bar(label: str, value: float, color: str = "#3b82f6"):
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="margin: 4px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#ccc;">
                <span>{label}</span><span>{pct}%</span>
            </div>
            <div class="quality-bar-bg">
                <div class="quality-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    model_path = st.text_input(
        "Model Checkpoint", value="checkpoints/best_model.pth"
    )
    temp_path = st.text_input(
        "Temperature File (optional)", value="checkpoints/temperature.pth"
    )

    st.markdown("### Thresholds")
    q_thresh = st.slider("Quality Threshold",    0.0, 1.0, 0.5, 0.05)
    c_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    e_thresh = st.slider("Entropy Threshold",    0.0, 3.0, 1.5, 0.1)

    st.markdown("---")
    mode = st.radio("Mode", ["📷 Upload Image", "📹 Webcam (Live)"])


# ─── Main UI ──────────────────────────────────────────────────────────────────

st.title("😊 Quality-Aware Real-Time FER")
st.caption("Facial Expression Recognition dengan Face Quality Assessment, Confidence Calibration, dan Reject Option")
st.markdown("---")

# Check apakah model tersedia
model_exists = Path(model_path).exists()

if not model_exists:
    st.warning(
        f"⚠️ Model checkpoint tidak ditemukan: `{model_path}`\n\n"
        "Jalankan `python train.py` terlebih dahulu untuk melatih model."
    )

# ─── Upload Image Mode ────────────────────────────────────────────────────────
if mode == "📷 Upload Image":
    uploaded = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])

    if uploaded and model_exists:
        col_img, col_result = st.columns([1, 1])

        # Load image
        pil_img  = Image.open(uploaded).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        pipeline = load_pipeline(model_path, temp_path)
        # Update thresholds
        pipeline["rejector"].q_threshold = q_thresh
        pipeline["rejector"].c_threshold = c_thresh
        pipeline["rejector"].e_threshold = e_thresh
        pipeline["quality"].q_threshold  = q_thresh

        with st.spinner("🔍 Analyzing..."):
            t0 = time.time()
            result, q_report, bbox = run_inference(pipeline, img_bgr)
            elapsed = time.time() - t0

        with col_img:
            st.subheader("Input Image")
            if result is not None:
                annotated = draw_bbox_on_image(img_bgr, bbox, result)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(pil_img, use_container_width=True)
                st.error("❌ Tidak ada wajah terdeteksi dalam gambar.")

        with col_result:
            st.subheader("Prediction Result")

            if result is None:
                st.error("Tidak ada wajah terdeteksi.")
            else:
                # Status badge
                status_class = "status-accepted" if result.is_accepted else "status-rejected"
                status_text  = "✅ ACCEPTED" if result.is_accepted else "❌ REJECTED"
                st.markdown(
                    f'<div class="metric-card"><span class="emotion-badge {status_class}">{status_text}</span></div>',
                    unsafe_allow_html=True,
                )

                if result.is_accepted and result.emotion:
                    emoji = EMOTION_EMOJIS.get(result.emotion, "")
                    color = EMOTION_COLORS.get(result.emotion, "#888")
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color:{color}">{emoji} {result.emotion}</div>'
                        f'<div class="metric-label">Predicted Emotion</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif not result.is_accepted:
                    st.error(f"Rejection reason: `{result.rejection_reason}`")

                # Metrics row
                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{result.confidence:.1%}")
                c2.metric("Entropy",    f"{result.entropy:.3f}")
                c3.metric("Latency",    f"{elapsed*1000:.0f}ms")

                st.markdown("#### Quality Scores")
                if q_report:
                    bar_colors = {
                        "blur":       "#3b82f6",
                        "brightness": "#f59e0b",
                        "pose":       "#8b5cf6",
                        "occlusion":  "#ec4899",
                        "face_size":  "#10b981",
                    }
                    for key, lbl in [
                        ("blur", "Blur"), ("brightness", "Brightness"),
                        ("pose", "Pose"), ("occlusion", "Occlusion"),
                        ("face_size", "Face Size"),
                    ]:
                        render_quality_bar(lbl, q_report.as_dict().get(key, 0.0), bar_colors[key])

                    q_total = q_report.quality_score
                    q_color = "#22c55e" if q_total >= q_thresh else "#ef4444"
                    st.markdown(f"**Total Quality Score:** <span style='color:{q_color};font-size:1.1rem'>{q_total:.2f}</span>", unsafe_allow_html=True)

                st.markdown("#### Class Probabilities")
                if result.probs:
                    for emotion, prob in sorted(result.probs.items(), key=lambda x: -x[1]):
                        render_quality_bar(
                            f"{EMOTION_EMOJIS.get(emotion,'')} {emotion}",
                            prob,
                            EMOTION_COLORS.get(emotion, "#888"),
                        )

# ─── Webcam Mode ──────────────────────────────────────────────────────────────
elif mode == "📹 Webcam (Live)":
    st.info("💡 Mode webcam live memerlukan **demo.py**. Klik tombol di bawah untuk memulai di terminal.")
    st.code(
        f"python demo.py \\\n"
        f"  --checkpoint {model_path} \\\n"
        f"  --temperature {temp_path} \\\n"
        f"  --q_threshold {q_thresh} \\\n"
        f"  --c_threshold {c_thresh} \\\n"
        f"  --e_threshold {e_thresh}",
        language="bash",
    )

    st.markdown("---")
    st.subheader("📸 Snapshot Mode (via Streamlit)")
    enable_cam = st.toggle("Aktifkan Kamera")

    if enable_cam and model_exists:
        img_file = st.camera_input("Ambil foto dari kamera")
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            pipeline = load_pipeline(model_path, temp_path)
            pipeline["rejector"].q_threshold = q_thresh
            pipeline["rejector"].c_threshold = c_thresh
            pipeline["rejector"].e_threshold = e_thresh
            pipeline["quality"].q_threshold  = q_thresh

            with st.spinner("Analyzing..."):
                result, q_report, bbox = run_inference(pipeline, img_bgr)

            if result is None:
                st.error("❌ Tidak ada wajah terdeteksi.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    annotated = draw_bbox_on_image(img_bgr, bbox, result)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    if result.is_accepted:
                        emoji = EMOTION_EMOJIS.get(result.emotion, "")
                        st.success(f"**{emoji} {result.emotion}** — {result.confidence:.1%} confidence")
                    else:
                        st.error(f"REJECTED: {result.rejection_reason}")

                    if q_report:
                        st.metric("Quality Score", f"{q_report.quality_score:.2f}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Quality-Aware Real-Time FER | EfficientNet-B0 + Temperature Scaling + Reject Option | 2026")
