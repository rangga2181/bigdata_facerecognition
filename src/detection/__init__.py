"""
src/detection/__init__.py
Factory untuk face detector.
"""
from .mediapipe_detector import MediaPipeDetector
from .retinaface_detector import RetinaFaceDetector


def get_detector(method: str = "mediapipe", **kwargs):
    """
    Factory method untuk memilih face detector.

    Args:
        method: 'mediapipe' atau 'retinaface'
        **kwargs: parameter tambahan untuk detector

    Returns:
        Face detector instance
    """
    method = method.lower()
    if method == "mediapipe":
        return MediaPipeDetector(**kwargs)
    elif method == "retinaface":
        return RetinaFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector method: {method}. Choose 'mediapipe' or 'retinaface'.")


__all__ = ["get_detector", "MediaPipeDetector", "RetinaFaceDetector"]
