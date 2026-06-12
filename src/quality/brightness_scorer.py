"""
src/quality/brightness_scorer.py
Menilai kualitas pencahayaan berdasarkan mean luminance.
"""
import cv2
import numpy as np


class BrightnessScorer:
    """
    Menilai kualitas pencahayaan frame.

    Score = 1.0 jika luminance dalam range optimal [low_opt, high_opt].
    Score menurun secara linear jika di luar range tersebut.

    Args:
        low_thresh:  batas bawah luminance (terlalu gelap < low_thresh)
        high_thresh: batas atas luminance (terlalu terang > high_thresh)
        low_opt:     batas bawah range optimal
        high_opt:    batas atas range optimal
    """

    def __init__(
        self,
        low_thresh: float = 20.0,
        high_thresh: float = 235.0,
        low_opt: float = 60.0,
        high_opt: float = 180.0,
    ):
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh
        self.low_opt     = low_opt
        self.high_opt    = high_opt

    def mean_luminance(self, image: np.ndarray) -> float:
        """
        Menghitung mean luminance channel Y dari ruang warna YCrCb.

        Args:
            image: BGR numpy array

        Returns:
            Mean luminance [0, 255]
        """
        if len(image.shape) == 2:
            return float(image.mean())
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return float(ycrcb[:, :, 0].mean())

    def score(self, image: np.ndarray) -> float:
        """
        Menghitung brightness score [0.0, 1.0].

        Returns:
            1.0 = pencahayaan optimal
            0.0 = terlalu gelap atau terlalu terang
        """
        lum = self.mean_luminance(image)

        if lum < self.low_thresh or lum > self.high_thresh:
            return 0.0
        elif self.low_opt <= lum <= self.high_opt:
            return 1.0
        elif lum < self.low_opt:
            # Linear dari low_thresh → low_opt
            return (lum - self.low_thresh) / (self.low_opt - self.low_thresh)
        else:
            # Linear dari high_opt → high_thresh
            return (self.high_thresh - lum) / (self.high_thresh - self.high_opt)
