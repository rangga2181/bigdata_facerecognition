"""
evaluate.py
Entry point untuk evaluasi lengkap model FER.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/fer2013.yaml
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

from src.fer.model import FERModel
from src.fer.dataset import FERDataset, get_dataloader
from src.fer.transforms import get_transforms
from src.calibration.temperature_scaling import TemperatureScaling
from src.calibration.calibration_evaluator import CalibrationEvaluator
from src.rejection.reject_option import RejectOption
from src.utils.metrics import compute_metrics, compute_classification_report, plot_confusion_matrix, plot_coverage_vs_risk
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FER Model")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--config",       type=str, default="configs/fer2013.yaml")
    parser.add_argument("--csv",          type=str, default=None)
    parser.add_argument("--temperature",  type=str, default=None, help="Path ke temperature.pth")
    parser.add_argument("--split",        type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device",       type=str, default=None)
    parser.add_argument("--output_dir",   type=str, default="results")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = get_logger("evaluate")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/plots").mkdir(parents=True, exist_ok=True)

    # ── Load model
    model = FERModel.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # ── Load calibrator
    calibrator = None
    if args.temperature and Path(args.temperature).exists():
        calibrator = TemperatureScaling.load(args.temperature)
        logger.info(f"Temperature loaded: T={calibrator.temperature.item():.4f}")

    # ── Dataset
    tf = get_transforms(args.split, image_size=cfg.dataset.image_size)
    ds = FERDataset(root=cfg.dataset.root, split=args.split, transform=tf, csv_file=args.csv)
    loader = get_dataloader(ds, batch_size=64, shuffle=False, num_workers=2)
    logger.info(f"{args.split.capitalize()} samples: {len(ds)}")

    # ── Inference
    all_logits  = []
    all_probs   = []
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)

            if calibrator:
                probs = calibrator.predict_proba(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            preds = probs.argmax(dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits  = np.concatenate(all_logits)
    all_probs   = np.concatenate(all_probs)
    all_preds   = np.concatenate(all_preds)
    all_labels  = np.concatenate(all_labels)

    # ── Classification metrics
    cls_metrics = compute_metrics(all_preds, all_labels)
    logger.info("=== Classification Metrics ===")
    for k, v in cls_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Classification report
    report = compute_classification_report(all_preds, all_labels)
    logger.info(f"\n{report}")

    # ── Calibration metrics
    cal_eval   = CalibrationEvaluator()
    cal_metrics = cal_eval.compute_all(all_probs, all_labels)
    logger.info("=== Calibration Metrics ===")
    for k, v in cal_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Selective prediction metrics (coverage-risk curve)
    rejector = RejectOption(
        q_threshold=cfg.rejection.c_threshold,   # pakai confidence threshold saja
        c_threshold=cfg.rejection.c_threshold,
        e_threshold=cfg.rejection.e_threshold,
    )

    coverages = []
    risks     = []
    for c_thresh in np.arange(0.3, 0.99, 0.05):
        rejector.c_threshold = round(float(c_thresh), 2)
        # Simulasikan quality score = 1.0 (assume semua frame lulus quality)
        fake_q = np.ones(len(all_probs))
        cov  = rejector.compute_coverage(all_probs, fake_q)
        risk = rejector.compute_selective_risk(all_probs, fake_q, all_labels)
        coverages.append(cov)
        risks.append(risk)

    logger.info(f"Coverage range: [{min(coverages):.2f}, {max(coverages):.2f}]")
    logger.info(f"Risk range:     [{min(risks):.2f}, {max(risks):.2f}]")

    # ── Plots
    plot_confusion_matrix(
        all_preds, all_labels,
        save_path=f"{args.output_dir}/plots/confusion_matrix.png"
    )
    plot_coverage_vs_risk(
        coverages, risks,
        save_path=f"{args.output_dir}/plots/coverage_vs_risk.png"
    )

    # ── Save summary JSON
    summary = {
        "checkpoint":  args.checkpoint,
        "split":       args.split,
        "n_samples":   len(ds),
        "classification": cls_metrics,
        "calibration":    cal_metrics,
    }
    summary_path = f"{args.output_dir}/reports/evaluation_summary.json"
    Path(f"{args.output_dir}/reports").mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
