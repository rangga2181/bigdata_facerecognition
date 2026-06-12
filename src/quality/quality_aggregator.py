"""
src/quality/quality_aggregator.py
Menggabungkan semua sub-scorer menjadi satu quality score final.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .blur_detector import BlurDetector
from .brightness_scorer import BrightnessScorer
from .pose_estimator import PoseEstimator
from .occlusion_estimator import OcclusionEstimator


@dataclass
class QualityReport:
    """Laporan kualitas frame per sub-komponen."""
    blur_score:        float = 0.0
    brightness_score:  float = 0.0
    pose_score:        float = 0.0
    occlusion_score:   float = 0.0
    face_size_score:   float = 0.0
    quality_score:     float = 0.0   # weighted aggregate
    accepted:          bool  = False
    rejection_reason:  str   = ""

    def as_dict(self) -> Dict[str, float]:
        return {
            "blur":        self.blur_score,
            "brightness":  self.brightness_score,
            "pose":        self.pose_score,
            "occlusion":   self.occlusion_score,
            "face_size":   self.face_size_score,
            "quality":     self.quality_score,
        }


class QualityAggregator:
    """
    Menggabungkan semua sub-scorer menjadi satu quality score.

    Args:
        weights: dict bobot per komponen (harus sum = 1.0)
        q_threshold: threshold quality score untuk penerimaan
        min_face_size: ukuran minimum wajah dalam pixel
    """

    _DEFAULT_WEIGHTS = {
        "blur":       0.30,
        "brightness": 0.20,
        "pose":       0.25,
        "occlusion":  0.15,
        "face_size":  0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        q_threshold: float = 0.5,
        min_face_size: int = 64,
    ):
        self.weights       = weights or self._DEFAULT_WEIGHTS
        self.q_threshold   = q_threshold
        self.min_face_size = min_face_size

        # Init sub-scorers
        self._blur       = BlurDetector()
        self._brightness = BrightnessScorer()
        self._pose       = PoseEstimator()
        self._occlusion  = OcclusionEstimator()

    def _face_size_score(self, bbox: Optional[Tuple[int, int, int, int]], frame_shape: Tuple) -> float:
        """Menghitung score berdasarkan ukuran wajah relatif terhadap frame."""
        if bbox is None:
            return 0.0
        _, _, fw, fh = bbox
        face_area  = fw * fh
        frame_area = frame_shape[0] * frame_shape[1]
        min_area   = self.min_face_size ** 2
        # Normalize: score 1.0 jika face area >= 10% frame
        target_area = frame_area * 0.10
        score = min(1.0, face_area / target_area)
        # Juga penalize jika terlalu kecil dari min_face_size
        if face_area < min_area:
            score *= face_area / min_area
        return float(score)

    def assess(
        self,
        face_image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple] = None,
    ) -> QualityReport:
        """
        Menilai kualitas face image.

        Args:
            face_image: cropped face BGR numpy array
            bbox: bounding box wajah dalam frame asli (opsional)
            frame_shape: shape frame asli untuk face size score

        Returns:
            QualityReport dengan semua sub-score dan final score
        """
        report = QualityReport()

        # Hitung sub-scores
        report.blur_score       = self._blur.score(face_image)
        report.brightness_score = self._brightness.score(face_image)
        report.pose_score       = self._pose.score(face_image)
        report.occlusion_score  = self._occlusion.score(face_image)

        if bbox is not None and frame_shape is not None:
            report.face_size_score = self._face_size_score(bbox, frame_shape)
        else:
            report.face_size_score = 1.0  # assume OK jika tidak ada info

        # Weighted aggregate
        w = self.weights
        report.quality_score = (
            w["blur"]       * report.blur_score       +
            w["brightness"] * report.brightness_score +
            w["pose"]       * report.pose_score       +
            w["occlusion"]  * report.occlusion_score  +
            w["face_size"]  * report.face_size_score
        )

        # Evaluasi penerimaan
        report.accepted = report.quality_score >= self.q_threshold

        if not report.accepted:
            # Tentukan penyebab utama penolakan
            scores = {
                "blur":       report.blur_score,
                "brightness": report.brightness_score,
                "pose":       report.pose_score,
                "occlusion":  report.occlusion_score,
                "face_size":  report.face_size_score,
            }
            worst = min(scores, key=scores.get)
            report.rejection_reason = f"low_{worst}"

        return report

    def close(self):
        self._pose.close()
        self._occlusion.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
