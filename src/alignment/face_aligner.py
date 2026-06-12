"""
src/alignment/face_aligner.py
Face alignment menggunakan eye landmark + affine transformation.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class FaceAligner:
    """
    Melakukan alignment wajah berdasarkan posisi mata.

    Pipeline:
        1. Deteksi 468 facial landmark via MediaPipe FaceMesh
        2. Ambil koordinat mata kiri & kanan
        3. Hitung sudut rotasi dari garis antar mata
        4. Affine transform supaya garis mata horizontal
        5. Crop & resize ke output_size

    Args:
        output_size: ukuran output (W, H), default (224, 224)
        eye_percentage: persentase offset vertikal dari top crop
    """

    # Index MediaPipe FaceMesh untuk pusat mata
    _LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self, output_size: Tuple[int, int] = (224, 224), eye_percentage: float = 0.35):
        self.output_size = output_size
        self.eye_percentage = eye_percentage

        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _get_eye_centers(self, landmarks, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Menghitung pusat koordinat mata kiri dan kanan."""
        def _mean(idx_list):
            pts = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in idx_list],
                dtype=np.float32,
            )
            return pts.mean(axis=0)

        left_center  = _mean(self._LEFT_EYE_IDX)
        right_center = _mean(self._RIGHT_EYE_IDX)
        return left_center, right_center

    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Align wajah dalam frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            Aligned & cropped face sebagai numpy array (224, 224, 3),
            atau None jika tidak ada wajah terdeteksi.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        left_eye, right_eye = self._get_eye_centers(lm, h, w)

        # Hitung sudut rotasi
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Pusat rotasi = titik tengah antara kedua mata
        eye_center = ((left_eye + right_eye) / 2).astype(np.float32)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1.0)
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

        # Hitung bounding box crop berdasarkan jarak antar mata
        eye_dist = np.linalg.norm(right_eye - left_eye)
        crop_size = int(eye_dist / 0.45)  # empirical factor

        cx, cy = int(eye_center[0]), int(eye_center[1])
        x1 = cx - crop_size // 2
        y1 = cy - int(crop_size * self.eye_percentage)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Clamp
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = rotated[y1:y2, x1:x2]
        aligned = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)
        return aligned

    def align_from_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Align wajah dari bounding box yang sudah diketahui.

        Args:
            frame: BGR frame
            bbox: (x, y, w, h)
            padding: padding relatif terhadap bbox

        Returns:
            Aligned face array
        """
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]

        # Tambahkan padding
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        return self.align(face_roi)

    def simple_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.15,
    ) -> Optional[np.ndarray]:
        """
        Crop sederhana tanpa alignment (fallback).

        Args:
            frame: BGR frame
            bbox: (x, y, w, h)

        Returns:
            Cropped & resized face
        """
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]

        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
