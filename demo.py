"""
demo.py
Entry point untuk demo real-time webcam FER.

Usage:
    python demo.py --checkpoint checkpoints/best_model.pth
    python demo.py --checkpoint checkpoints/best_model.pth --temperature checkpoints/temperature.pth
    python demo.py --checkpoint checkpoints/best_model.pth --record results/demo.mp4
"""
import argparse
import sys

sys.path.insert(0, ".")

from src.realtime.webcam_pipeline import WebcamPipeline
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time FER Demo")
    parser.add_argument("--checkpoint",   type=str, required=True,          help="Path ke model checkpoint")
    parser.add_argument("--temperature",  type=str, default=None,           help="Path ke temperature.pth")
    parser.add_argument("--detector",     type=str, default="mediapipe",    help="mediapipe atau retinaface")
    parser.add_argument("--camera",       type=int, default=0,              help="ID kamera")
    parser.add_argument("--q_threshold",  type=float, default=0.5)
    parser.add_argument("--c_threshold",  type=float, default=0.7)
    parser.add_argument("--e_threshold",  type=float, default=1.5)
    parser.add_argument("--width",        type=int, default=1280)
    parser.add_argument("--height",       type=int, default=720)
    parser.add_argument("--record",       type=str, default=None,           help="Path output video recording")
    parser.add_argument("--device",       type=str, default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    logger = get_logger("demo")

    logger.info("🎥 Starting Real-Time FER Demo")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Detector:   {args.detector}")
    logger.info(f"  Thresholds: Q={args.q_threshold}, C={args.c_threshold}, E={args.e_threshold}")

    pipeline = WebcamPipeline(
        model_path=args.checkpoint,
        temperature_path=args.temperature,
        detector_method=args.detector,
        q_threshold=args.q_threshold,
        c_threshold=args.c_threshold,
        e_threshold=args.e_threshold,
        camera_id=args.camera,
        display_size=(args.width, args.height),
        device=args.device,
        record_output=args.record,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
