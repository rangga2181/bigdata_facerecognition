"""
src/quality/blur_detector.py
Mendeteksi tingkat blur menggunakan Laplacian variance.
"""
import cv2
import numpy as np


class BlurDetector:
    """
    Menghitung blur score berdasarkan Laplacian variance.

    Semakin tinggi variance → gambar lebih tajam → score mendekati 1.0
    Semakin rendah variance → gambar lebih blur → score mendekati 0.0

    Args:
        sharp_threshold: nilai variance yang dianggap "sangat tajam" (default=500)
    """

    def __init__(self, sharp_threshold: float = 500.0):
        self.sharp_threshold = sharp_threshold

    def compute_variance(self, image: np.ndarray) -> float:
        """
        Menghitung Laplacian variance dari image.

        Args:
            image: numpy array BGR atau grayscale

        Returns:
            Laplacian variance (semakin tinggi = semakin tajam)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def score(self, image: np.ndarray) -> float:
        """
        Menghitung blur score [0.0, 1.0].

        Returns:
            0.0 = sangat blur, 1.0 = sangat tajam
        """
        variance = self.compute_variance(image)
        # Normalisasi menggunakan clipping
        score = min(1.0, variance / self.sharp_threshold)
        return float(score)

    def is_blurry(self, image: np.ndarray, threshold: float = 0.3) -> bool:
        """Mengembalikan True jika image dianggap blur."""
        return self.score(image) < threshold
