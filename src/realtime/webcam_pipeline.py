"""
src/realtime/webcam_pipeline.py
End-to-end real-time FER pipeline dari webcam.
"""
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

from ..detection import get_detector
from ..alignment.face_aligner import FaceAligner
from ..quality.quality_aggregator import QualityAggregator
from ..fer.model import FERModel
from ..fer.transforms import get_inference_transform
from ..calibration.temperature_scaling import TemperatureScaling
from ..rejection.reject_option import RejectOption, PredictionResult
from .visualizer import Visualizer


class WebcamPipeline:
    """
    Pipeline FER real-time yang menghubungkan semua komponen:

        Webcam → Detect → Align → Quality → [Reject?]
               → FER Predict → Calibrate → [Reject?]
               → Visualize → Display

    Args:
        model_path:       path ke checkpoint model (.pth)
        temperature_path: path ke file temperature (opsional)
        detector_method:  'mediapipe' atau 'retinaface'
        q_threshold:      quality threshold
        c_threshold:      confidence threshold
        e_threshold:      entropy threshold
        camera_id:        ID kamera (default 0)
        display_size:     (width, height) untuk tampilan
        device:           'cuda' atau 'cpu'
        record_output:    path untuk menyimpan video recording (opsional)
    """

    def __init__(
        self,
        model_path: str,
        temperature_path: Optional[str] = None,
        detector_method: str = "mediapipe",
        q_threshold: float = 0.5,
        c_threshold: float = 0.7,
        e_threshold: float = 1.5,
        camera_id: int = 0,
        display_size: Tuple[int, int] = (1280, 720),
        device: Optional[str] = None,
        record_output: Optional[str] = None,
    ):
        self.camera_id    = camera_id
        self.display_size = display_size
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.record_output = record_output

        print(f"🚀 Initializing WebcamPipeline on {self.device.upper()}...")

        # ── Load model
        self.model = FERModel.load_from_checkpoint(model_path, device=self.device)
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")

        # ── Temperature scaling
        if temperature_path and Path(temperature_path).exists():
            self.calibrator = TemperatureScaling.load(temperature_path)
        else:
            self.calibrator = TemperatureScaling(temperature=1.0)
            print("⚠️  No temperature file found. Using T=1.0 (no scaling).")

        # ── Components
        self.detector   = get_detector(detector_method)
        self.aligner    = FaceAligner()
        self.quality    = QualityAggregator(q_threshold=q_threshold)
        self.rejector   = RejectOption(q_threshold=q_threshold, c_threshold=c_threshold, e_threshold=e_threshold)
        self.visualizer = Visualizer()
        self.transform  = get_inference_transform()

        print("✅ All components initialized.")

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """BGR numpy → tensor untuk inference."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor   = self.transform(face_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def _infer(self, face_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menjalankan inference dan mengembalikan (logits, calibrated_probs).
        """
        with torch.no_grad():
            logits = self.model(face_tensor)
            cal_probs = self.calibrator.predict_proba(logits)
        return logits.cpu().numpy()[0], cal_probs.cpu().numpy()[0]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Memproses satu frame dan mengembalikan frame ter-annotasi.

        Args:
            frame: BGR numpy array dari webcam

        Returns:
            Annotated BGR frame
        """
        # 1. Deteksi wajah
        detection = self.detector.detect_largest(frame)

        quality_report = {}
        result         = None
        bbox           = None

        if detection is not None:
            bbox = detection.bbox

            # 2. Align + crop
            face = self.aligner.align_from_bbox(frame, bbox)
            if face is None:
                face = self.aligner.simple_crop(frame, bbox)

            if face is not None:
                # 3. Quality assessment
                q_report = self.quality.assess(face, bbox=bbox, frame_shape=frame.shape[:2])
                quality_report = q_report.as_dict()

                if not q_report.accepted:
                    # Rejected karena kualitas rendah
                    result = PredictionResult(
                        status="rejected",
                        emotion=None,
                        emotion_idx=None,
                        confidence=0.0,
                        entropy=0.0,
                        quality_score=q_report.quality_score,
                        rejection_reason=q_report.rejection_reason,
                        probs={},
                    )
                else:
                    # 4. Inference
                    tensor = self._preprocess(face)
                    _, probs = self._infer(tensor)

                    # 5. Reject option
                    result = self.rejector.decide(probs, q_report.quality_score)

        return quality_report, result, bbox

    def run(self):
        """
        Menjalankan pipeline webcam secara loop.

        Keyboard shortcuts:
            q / ESC : keluar
            s       : screenshot
            p       : toggle prob panel
            r       : toggle recording
        """
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_size[1])

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Video writer
        writer = None
        recording = False
        if self.record_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.record_output, fourcc, 25.0,
                (self.display_size[0], self.display_size[1])
            )
            recording = True
            print(f"🎥 Recording to {self.record_output}")

        print("\n▶  Running webcam pipeline. Press 'q' or ESC to quit.\n")

        fps_counter = 0
        fps_display = 0.0
        t_fps_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame.")
                    continue

                # Proses frame
                quality_report, result, bbox = self.process_frame(frame)

                # FPS
                fps_counter += 1
                elapsed = time.time() - t_fps_start
                if elapsed >= 1.0:
                    fps_display   = fps_counter / elapsed
                    fps_counter   = 0
                    t_fps_start   = time.time()

                # Visualisasi
                annotated = self.visualizer.render(
                    frame, bbox, result, quality_report, fps=fps_display
                )

                # Recording indicator
                if recording:
                    cv2.circle(annotated, (annotated.shape[1] - 20, 20), 8, (0, 0, 220), -1)
                    if writer:
                        writer.write(annotated)

                cv2.imshow("Quality-Aware FER", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q atau ESC
                    break
                elif key == ord("s"):
                    ts = int(time.time())
                    path = f"results/screenshot_{ts}.png"
                    Path("results").mkdir(exist_ok=True)
                    cv2.imwrite(path, annotated)
                    print(f"📸 Screenshot saved: {path}")
                elif key == ord("p"):
                    self.visualizer.show_probs = not self.visualizer.show_probs

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.detector.close() if hasattr(self.detector, "close") else None
            self.aligner.close()
            self.quality.close()
            print("✅ Pipeline stopped.")
