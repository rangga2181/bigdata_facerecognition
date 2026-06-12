"""
src/detection/retinaface_detector.py
Face detection menggunakan RetinaFace via OpenCV DNN (fallback-safe).
Jika library retinaface tidak tersedia, menggunakan Haar Cascade sebagai fallback.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

try:
    from retinaface import RetinaFace as _RF
    _RETINAFACE_AVAILABLE = True
except ImportError:
    _RETINAFACE_AVAILABLE = False


@dataclass
class FaceDetection:
    """Hasil deteksi wajah."""
    bbox: tuple          # (x, y, w, h)
    confidence: float
    landmarks: Optional[np.ndarray] = None


class RetinaFaceDetector:
    """
    Face detector menggunakan RetinaFace.
    Jika library tidak tersedia, otomatis fallback ke OpenCV Haar Cascade.

    Args:
        threshold: confidence threshold
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        if not _RETINAFACE_AVAILABLE:
            print("[WARNING] RetinaFace not installed. Falling back to OpenCV Haar Cascade.")
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            self._cascade = None

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Mendeteksi wajah dalam frame BGR.

        Args:
            frame: numpy array BGR (H, W, 3)

        Returns:
            List of FaceDetection
        """
        if not _RETINAFACE_AVAILABLE:
            return self._detect_haar(frame)
        return self._detect_retinaface(frame)

    def _detect_retinaface(self, frame: np.ndarray) -> List[FaceDetection]:
        """Deteksi menggunakan library RetinaFace."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resp = _RF.detect_faces(rgb)

        detections: List[FaceDetection] = []
        if not isinstance(resp, dict):
            return detections

        for key, val in resp.items():
            score = val.get("score", 0.0)
            if score < self.threshold:
                continue

            fa = val["facial_area"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = fa
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            lm = val.get("landmarks", {})
            kps = []
            for part in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
                pt = lm.get(part)
                if pt:
                    kps.append(pt)
            landmarks = np.array(kps, dtype=np.float32) if kps else None

            detections.append(FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(score),
                landmarks=landmarks,
            ))

        return detections

    def _detect_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        """Fallback: deteksi menggunakan Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detections: List[FaceDetection] = []
        if len(faces) == 0:
            return detections

        for (x, y, w, h) in faces:
            detections.append(FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=1.0,
            ))

        return detections

    def detect_largest(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Mengembalikan wajah terbesar dalam frame."""
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
