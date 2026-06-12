"""
calibrate.py
Entry point untuk melakukan Temperature Scaling calibration.

Usage:
    python calibrate.py --checkpoint checkpoints/best_model.pth --config configs/fer2013.yaml
"""
import argparse
import sys
import torch

sys.path.insert(0, ".")

from src.fer.model import FERModel
from src.fer.dataset import FERDataset, get_dataloader
from src.fer.transforms import get_transforms
from src.calibration.temperature_scaling import TemperatureScaling
from src.calibration.calibration_evaluator import CalibrationEvaluator
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate FER Model using Temperature Scaling")
    parser.add_argument("--checkpoint", type=str, required=True,    help="Path ke best_model.pth")
    parser.add_argument("--config",     type=str, default="configs/fer2013.yaml")
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--output",     type=str, default="checkpoints/temperature.pth")
    return parser.parse_args()


def collect_probs(model, loader, device, calibrator=None):
    """Mengumpulkan probabilities dari loader."""
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            if calibrator:
                probs = calibrator.predict_proba(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = get_logger("calibrate")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model
    model = FERModel.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    logger.info(f"Model loaded: {args.checkpoint}")

    # ── Val dataset
    val_tf = get_transforms("val", image_size=cfg.dataset.image_size)
    val_ds = FERDataset(root=cfg.dataset.root, split="val", transform=val_tf, csv_file=args.csv)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    logger.info(f"Val samples: {len(val_ds)}")

    evaluator = CalibrationEvaluator(n_bins=15)

    # ── Before calibration
    probs_before, labels = collect_probs(model, val_loader, device)
    metrics_before = evaluator.compute_all(probs_before, labels)
    logger.info(f"Before calibration → ECE: {metrics_before['ECE']:.4f} | Brier: {metrics_before['BrierScore']:.4f}")

    # ── Calibrate
    ts = TemperatureScaling(temperature=1.0)
    ts.calibrate(model, val_loader, device=device, max_iter=50)
    ts.save(args.output)

    # ── After calibration
    probs_after, _ = collect_probs(model, val_loader, device, calibrator=ts)
    metrics_after  = evaluator.compute_all(probs_after, labels)
    logger.info(f"After calibration  → ECE: {metrics_after['ECE']:.4f} | Brier: {metrics_after['BrierScore']:.4f}")

    # ── Plot reliability diagram
    evaluator.plot_reliability_diagram(
        probs_before, probs_after, labels,
        save_path="results/plots/calibration_curve.png"
    )

    logger.info("Calibration complete!")


if __name__ == "__main__":
    main()
