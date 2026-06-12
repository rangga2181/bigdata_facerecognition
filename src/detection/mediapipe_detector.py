"""
src/detection/mediapipe_detector.py
Face detection menggunakan MediaPipe Face Detection.
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FaceDetection:
    """Hasil deteksi wajah."""
    bbox: tuple          # (x, y, w, h) dalam pixel
    confidence: float    # confidence score [0, 1]
    landmarks: Optional[np.ndarray] = None  # key points jika tersedia


class MediaPipeDetector:
    """
    Face detector menggunakan MediaPipe.

    Args:
        min_detection_confidence: threshold confidence (default 0.5)
        model_selection: 0 = short-range (<2m), 1 = full-range (<5m)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Mendeteksi wajah dalam frame BGR.

        Args:
            frame: numpy array BGR (H, W, 3)

        Returns:
            List of FaceDetection
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        detections: List[FaceDetection] = []
        if not results.detections:
            return detections

        for det in results.detections:
            bbox_rel = det.location_data.relative_bounding_box
            x = int(bbox_rel.xmin * w)
            y = int(bbox_rel.ymin * h)
            bw = int(bbox_rel.width * w)
            bh = int(bbox_rel.height * h)

            # Clamp ke batas frame
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # Key points (6 titik: right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear)
            kps = []
            for kp in det.location_data.relative_keypoints:
                kps.append([kp.x * w, kp.y * h])
            landmarks = np.array(kps, dtype=np.float32) if kps else None

            score = det.score[0] if det.score else 0.0

            detections.append(FaceDetection(
                bbox=(x, y, bw, bh),
                confidence=float(score),
                landmarks=landmarks,
            ))

        return detections

    def detect_largest(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """Mengembalikan wajah terbesar (area terluas) dalam frame."""
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: d.bbox[2] * d.bbox[3])

    def close(self):
        """Menutup resource MediaPipe."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
