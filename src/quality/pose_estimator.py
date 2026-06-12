"""
src/quality/pose_estimator.py
Estimasi pose kepala (yaw, pitch, roll) menggunakan MediaPipe FaceMesh.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional


class PoseEstimator:
    """
    Mengestimasi pose kepala (yaw, pitch, roll) dan menghasilkan pose score.

    Pose score = 1.0 jika kepala menghadap lurus ke depan.
    Score menurun bila ada rotasi yang berlebihan.

    Args:
        max_yaw:   batas yaw (derajat) yang masih diterima (default=30)
        max_pitch: batas pitch (derajat) yang masih diterima (default=20)
        max_roll:  batas roll (derajat) yang masih diterima (default=20)
    """

    # Model points 3D wajah (canonical)
    _MODEL_POINTS = np.array([
        (0.0,   0.0,    0.0),      # Nose tip
        (0.0,  -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),   # Left eye left corner
        (225.0,  170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0,  -150.0, -125.0),  # Right mouth corner
    ], dtype=np.float64)

    # Index landmark MediaPipe untuk 6 titik di atas
    _LM_IDX = [1, 152, 226, 446, 57, 287]

    def __init__(self, max_yaw: float = 30.0, max_pitch: float = 20.0, max_roll: float = 20.0):
        self.max_yaw   = max_yaw
        self.max_pitch = max_pitch
        self.max_roll  = max_roll

        self._mp_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Mengestimasi yaw, pitch, roll dari gambar.

        Args:
            image: BGR numpy array

        Returns:
            (yaw, pitch, roll) dalam derajat, atau None jika gagal
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # Ambil 6 image points
        img_points = np.array([
            [lm[i].x * w, lm[i].y * h] for i in self._LM_IDX
        ], dtype=np.float64)

        # Camera matrix (pinhole approximation)
        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        _, rvec, _ = cv2.solvePnP(
            self._MODEL_POINTS, img_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert rotation vector ke Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll  = 0.0

        return float(yaw), float(pitch), float(roll)

    def score(self, image: np.ndarray) -> float:
        """
        Menghitung pose score [0.0, 1.0].

        Returns:
            1.0 = kepala menghadap lurus,
            0.0 = pose terlalu miring
        """
        angles = self.estimate(image)
        if angles is None:
            return 0.0

        yaw, pitch, roll = angles

        # Hitung score per komponen (1 - normalized_error)
        yaw_score   = max(0.0, 1.0 - abs(yaw)   / self.max_yaw)
        pitch_score = max(0.0, 1.0 - abs(pitch) / self.max_pitch)
        roll_score  = max(0.0, 1.0 - abs(roll)  / self.max_roll)

        return float((yaw_score + pitch_score + roll_score) / 3.0)

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
