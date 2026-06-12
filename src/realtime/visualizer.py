"""
src/realtime/visualizer.py
Overlay UI untuk menampilkan hasil prediksi di frame webcam.
"""
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

# Warna per status dan emosi
STATUS_COLORS = {
    "accepted": (0, 220, 100),   # hijau
    "rejected": (0, 80, 220),    # merah-oranye (BGR)
}

EMOTION_COLORS = {
    "Angry":    (0, 0, 230),
    "Disgust":  (0, 140, 200),
    "Fear":     (180, 0, 200),
    "Happy":    (0, 210, 80),
    "Sad":      (220, 80, 0),
    "Surprise": (0, 180, 255),
    "Neutral":  (180, 180, 180),
}


class Visualizer:
    """
    Menambahkan overlay informasi ke frame webcam.

    Overlay mencakup:
        - Bounding box wajah (warna berdasarkan status)
        - Label emosi + confidence
        - Quality score bar
        - Confidence bar per emosi
        - FPS counter
        - Status "ACCEPTED" / "REJECTED" + reason
    """

    def __init__(self, show_probs: bool = True, show_quality: bool = True):
        self.show_probs   = show_probs
        self.show_quality = show_quality

        # Font settings
        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.55
        self._thickness  = 1

    # ─── Internal helpers ──────────────────────────────────────────────

    def _put_text(self, frame, text: str, pos: Tuple[int, int], color=(255, 255, 255), scale=None, thickness=None):
        scale = scale or self._font_scale
        thick = thickness or self._thickness
        # Shadow
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1), self._font, scale, (0,0,0), thick+1, cv2.LINE_AA)
        cv2.putText(frame, text, pos, self._font, scale, color, thick, cv2.LINE_AA)

    def _draw_bar(self, frame, label: str, value: float, pos: Tuple[int, int], width: int = 140, color=(0, 200, 100)):
        x, y = pos
        bar_h = 12

        # Background bar
        cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (50, 50, 50), -1)
        # Filled bar
        fill = int(width * min(1.0, max(0.0, value)))
        cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), color, -1)
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (120, 120, 120), 1)
        # Label + value text
        self._put_text(frame, f"{label}: {value:.2f}", (x, y - 3), color=(210, 210, 210), scale=0.40)

    # ─── Public methods ────────────────────────────────────────────────

    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        status: str = "accepted",
        thickness: int = 2,
    ):
        """Menggambar bounding box wajah."""
        x, y, w, h = bbox
        color = STATUS_COLORS.get(status, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    def draw_emotion_label(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        emotion: Optional[str],
        confidence: float,
        status: str,
    ):
        """Menggambar label emosi di atas bounding box."""
        x, y, w, h = bbox
        color = STATUS_COLORS.get(status, (255, 255, 255))

        if emotion:
            em_color = EMOTION_COLORS.get(emotion, (200, 200, 200))
            text = f"{emotion} ({confidence:.0%})"
        else:
            em_color = (0, 80, 220)
            text = "REJECTED"

        # Background label
        (tw, th), _ = cv2.getTextSize(text, self._font, self._font_scale, self._thickness)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
        self._put_text(frame, text, (x + 4, y - 5), color=em_color)

    def draw_quality_panel(
        self,
        frame: np.ndarray,
        quality_report: Dict[str, float],
        origin: Tuple[int, int] = (10, 10),
    ):
        """Menggambar panel quality scores di pojok kiri atas."""
        x, y = origin
        panel_w = 200
        panel_h = 130

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 5), (x + panel_w, y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        self._put_text(frame, "Frame Quality", (x, y + 12), color=(255, 215, 0), scale=0.55)

        keys = ["blur", "brightness", "pose", "occlusion", "face_size"]
        labels = ["Blur", "Brightness", "Pose", "Occlusion", "Face Size"]
        colors = [(100, 200, 100), (100, 200, 255), (200, 150, 100), (200, 100, 200), (150, 200, 200)]

        for i, (key, lbl, col) in enumerate(zip(keys, labels, colors)):
            val = quality_report.get(key, 0.0)
            self._draw_bar(frame, lbl, val, (x, y + 25 + i * 22), width=panel_w - 10, color=col)

        # Quality total
        q = quality_report.get("quality", 0.0)
        q_color = (0, 200, 100) if q >= 0.5 else (0, 80, 220)
        self._put_text(frame, f"Total: {q:.2f}", (x, y + 125), color=q_color, scale=0.55)

    def draw_probs_panel(
        self,
        frame: np.ndarray,
        probs: Dict[str, float],
        origin: Tuple[int, int] = None,
    ):
        """Menggambar panel confidence per emosi di pojok kanan atas."""
        if not probs:
            return

        h, w = frame.shape[:2]
        panel_w = 180
        panel_h = 170
        x = origin[0] if origin else w - panel_w - 15
        y = origin[1] if origin else 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 5), (x + panel_w, y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        self._put_text(frame, "Emotion Probs", (x, y + 12), color=(255, 215, 0), scale=0.55)

        for i, (emotion, prob) in enumerate(probs.items()):
            col = EMOTION_COLORS.get(emotion, (180, 180, 180))
            self._draw_bar(frame, emotion, prob, (x, y + 25 + i * 22), width=panel_w - 10, color=col)

    def draw_rejection_reason(
        self,
        frame: np.ndarray,
        reason: str,
        bbox: Tuple[int, int, int, int],
    ):
        """Menampilkan alasan penolakan di bawah bounding box."""
        x, y, w, h = bbox
        text = f"Rejected: {reason}"
        self._put_text(frame, text, (x, y + h + 18), color=(50, 100, 255), scale=0.45)

    def draw_fps(self, frame: np.ndarray, fps: float, pos: Tuple[int, int] = None):
        """Menampilkan FPS counter."""
        h, fw = frame.shape[:2]
        pos = pos or (fw - 100, h - 15)
        color = (0, 220, 120) if fps >= 20 else (50, 100, 255)
        self._put_text(frame, f"FPS: {fps:.1f}", pos, color=color, scale=0.55)

    def render(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        prediction_result,        # PredictionResult dari reject_option
        quality_report: Dict,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Render semua overlay ke frame.

        Args:
            frame:             BGR frame
            bbox:              bounding box wajah
            prediction_result: PredictionResult instance
            quality_report:    dict quality scores
            fps:               frames per second

        Returns:
            Annotated frame
        """
        out = frame.copy()

        if bbox is not None and prediction_result is not None:
            status   = prediction_result.status
            emotion  = prediction_result.emotion
            conf     = prediction_result.confidence
            reason   = prediction_result.rejection_reason
            probs    = prediction_result.probs

            self.draw_bbox(out, bbox, status=status)
            self.draw_emotion_label(out, bbox, emotion, conf, status)

            if status == "rejected" and reason:
                self.draw_rejection_reason(out, reason, bbox)

            if self.show_probs and probs:
                self.draw_probs_panel(out, probs)

        if self.show_quality and quality_report:
            self.draw_quality_panel(out, quality_report)

        self.draw_fps(out, fps)
        return out
