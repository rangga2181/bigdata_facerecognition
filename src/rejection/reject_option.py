"""
src/rejection/reject_option.py
Implementasi Reject Option berdasarkan triple-threshold mechanism.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List


EMOTION_CLASSES = {
    0: "angry",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "suprise",
}


@dataclass
class PredictionResult:
    """Hasil akhir prediksi setelah reject option."""
    status:           str            # "accepted" atau "rejected"
    emotion:          Optional[str]  # label emosi jika accepted
    emotion_idx:      Optional[int]  # index emosi jika accepted
    confidence:       float          # max confidence
    entropy:          float          # prediction entropy
    quality_score:    float          # frame quality score
    rejection_reason: str            # alasan penolakan jika rejected
    probs:            Dict[str, float] = field(default_factory=dict)  # semua class probs

    @property
    def is_accepted(self) -> bool:
        return self.status == "accepted"

    def as_dict(self) -> dict:
        return {
            "status":           self.status,
            "emotion":          self.emotion,
            "confidence":       round(self.confidence, 4),
            "entropy":          round(self.entropy, 4),
            "quality_score":    round(self.quality_score, 4),
            "rejection_reason": self.rejection_reason,
            "probs":            {k: round(v, 4) for k, v in self.probs.items()},
        }


class RejectOption:
    """
    Menentukan apakah prediksi diterima atau ditolak berdasarkan
    tiga kriteria:
        1. Quality threshold (kualitas frame)
        2. Confidence threshold (max confidence)
        3. Entropy threshold (ketidakpastian prediksi)

    Args:
        q_threshold: quality score minimum (default=0.5)
        c_threshold: confidence minimum (default=0.7)
        e_threshold: entropy maksimum (default=1.5)
    """

    def __init__(
        self,
        q_threshold: float = 0.5,
        c_threshold: float = 0.7,
        e_threshold: float = 1.5,
    ):
        self.q_threshold = q_threshold
        self.c_threshold = c_threshold
        self.e_threshold = e_threshold

    @staticmethod
    def compute_entropy(probs: np.ndarray) -> float:
        """
        Menghitung Shannon entropy dari distribusi probabilitas.

        H = -sum(p * log(p))

        Args:
            probs: array probabilitas (C,) yang sudah di-softmax

        Returns:
            Entropy dalam nats
        """
        probs = np.clip(probs, 1e-10, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def decide(
        self,
        probs: np.ndarray,
        quality_score: float,
    ) -> PredictionResult:
        """
        Memutuskan apakah prediksi diterima atau ditolak.

        Args:
            probs:         array softmax probabilities (C,)
            quality_score: quality score frame [0, 1]

        Returns:
            PredictionResult dengan status, alasan, dan label emosi
        """
        confidence = float(probs.max())
        entropy    = self.compute_entropy(probs)
        pred_idx   = int(probs.argmax())
        pred_label = EMOTION_CLASSES.get(pred_idx, f"class_{pred_idx}")

        # Build probs dict
        probs_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(probs))}

        # ─── Decision Tree ───────────────────────────────────────
        rejection_reason = ""

        if quality_score < self.q_threshold:
            rejection_reason = f"low_quality ({quality_score:.2f} < {self.q_threshold})"
        elif confidence < self.c_threshold:
            rejection_reason = f"low_confidence ({confidence:.2f} < {self.c_threshold})"
        elif entropy > self.e_threshold:
            rejection_reason = f"high_entropy ({entropy:.2f} > {self.e_threshold})"
        # ─────────────────────────────────────────────────────────

        if rejection_reason:
            return PredictionResult(
                status="rejected",
                emotion=None,
                emotion_idx=None,
                confidence=confidence,
                entropy=entropy,
                quality_score=quality_score,
                rejection_reason=rejection_reason,
                probs=probs_dict,
            )
        else:
            return PredictionResult(
                status="accepted",
                emotion=pred_label,
                emotion_idx=pred_idx,
                confidence=confidence,
                entropy=entropy,
                quality_score=quality_score,
                rejection_reason="",
                probs=probs_dict,
            )

    def compute_coverage(
        self,
        all_probs: np.ndarray,
        all_quality_scores: np.ndarray,
    ) -> float:
        """
        Menghitung coverage (rasio prediksi yang diterima).

        Args:
            all_probs:          array (N, C)
            all_quality_scores: array (N,)

        Returns:
            Coverage [0, 1]
        """
        accepted = sum(
            1 for i in range(len(all_probs))
            if self.decide(all_probs[i], all_quality_scores[i]).is_accepted
        )
        return accepted / len(all_probs)

    def compute_selective_risk(
        self,
        all_probs: np.ndarray,
        all_quality_scores: np.ndarray,
        all_labels: np.ndarray,
    ) -> float:
        """
        Menghitung selective risk (error rate pada prediksi yang diterima).

        Args:
            all_probs:          array (N, C)
            all_quality_scores: array (N,)
            all_labels:         array (N,) ground truth

        Returns:
            Selective risk [0, 1]
        """
        correct = 0
        accepted = 0
        for i in range(len(all_probs)):
            result = self.decide(all_probs[i], all_quality_scores[i])
            if result.is_accepted:
                accepted += 1
                if result.emotion_idx == all_labels[i]:
                    correct += 1
        if accepted == 0:
            return 1.0
        return 1.0 - correct / accepted

    def tune_thresholds(
        self,
        all_probs: np.ndarray,
        all_quality_scores: np.ndarray,
        all_labels: np.ndarray,
        target_coverage: float = 0.8,
    ) -> Dict[str, float]:
        """
        Auto-tune thresholds untuk mencapai target coverage.

        Args:
            all_probs:       (N, C)
            all_quality_scores: (N,)
            all_labels:      (N,)
            target_coverage: target coverage yang diinginkan

        Returns:
            Dict dengan thresholds optimal
        """
        best_risk = 1.0
        best_cfg  = {"c_threshold": self.c_threshold, "e_threshold": self.e_threshold}

        for c in np.arange(0.5, 0.95, 0.05):
            for e in np.arange(0.5, 2.5, 0.1):
                self.c_threshold = round(float(c), 2)
                self.e_threshold = round(float(e), 2)
                cov  = self.compute_coverage(all_probs, all_quality_scores)
                risk = self.compute_selective_risk(all_probs, all_quality_scores, all_labels)
                if cov >= target_coverage and risk < best_risk:
                    best_risk = risk
                    best_cfg  = {
                        "c_threshold": self.c_threshold,
                        "e_threshold": self.e_threshold,
                    }

        # Apply best config
        self.c_threshold = best_cfg["c_threshold"]
        self.e_threshold = best_cfg["e_threshold"]
        print(f"✅ Best thresholds: c={self.c_threshold}, e={self.e_threshold}, risk={best_risk:.4f}")
        return best_cfg
