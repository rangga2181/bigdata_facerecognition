"""
src/quality/occlusion_estimator.py
Estimasi occlusion berdasarkan visibilitas landmark wajah.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


class OcclusionEstimator:
    """
    Mengestimasi tingkat occlusion berdasarkan visibilitas landmark MediaPipe.

    Occlusion score = 1.0 → tidak ada occlusion
    Occlusion score = 0.0 → wajah sangat tertutup

    Args:
        key_regions: indeks landmark yang diperiksa (default: area mulut, mata, hidung)
    """

    # Landmark kunci: mata kiri, mata kanan, hidung, bibir atas, bibir bawah, pipi
    _KEY_LANDMARKS = [
        # Mata kiri
        33, 160, 158, 133, 153, 144,
        # Mata kanan
        362, 385, 387, 263, 373, 380,
        # Hidung
        1, 2, 5,
        # Bibir
        61, 291, 13, 14,
        # Dahi
        10, 9,
        # Dagu
        152,
    ]

    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def score(self, image: np.ndarray) -> float:
        """
        Menghitung occlusion score [0.0, 1.0].

        Returns:
            1.0 = tidak ada occlusion,
            0.0 = sangat tertutup
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return 0.0

        lm = results.multi_face_landmarks[0].landmark
        total = len(self._KEY_LANDMARKS)
        visible = 0

        for idx in self._KEY_LANDMARKS:
            if idx < len(lm):
                point = lm[idx]
                # Landmark dalam frame (x,y dalam [0,1])
                if 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0:
                    visible += 1

        return float(visible / total)

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
