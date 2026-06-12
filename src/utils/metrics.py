"""
src/utils/metrics.py
Semua metrik evaluasi: classification, calibration, selective prediction.
"""
import numpy as np
import matplotlib
# Hanya set Agg backend jika tidak dijalankan dari Jupyter/IPython
is_jupyter = False
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        is_jupyter = True
except ImportError:
    pass

if not is_jupyter:
    matplotlib.use("Agg")  # script mode: simpan file saja
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

EMOTION_CLASSES = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Menghitung classification metrics.

    Args:
        preds:   predicted class indices (N,)
        labels:  ground truth class indices (N,)
        average: averaging strategy

    Returns:
        Dict dengan accuracy, precision, recall, f1, uar
    """
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average=average, zero_division=0)
    rec  = recall_score(labels, preds, average=average, zero_division=0)
    f1   = f1_score(labels, preds, average=average, zero_division=0)

    # UAR = Unweighted Average Recall (macro recall)
    uar = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "UAR":       float(uar),
    }


def compute_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Mengembalikan classification report string."""
    names = class_names or EMOTION_CLASSES
    return classification_report(labels, preds, target_names=names, zero_division=0)


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "results/plots/confusion_matrix.png",
    normalize: bool = True,
):
    """
    Membuat dan menyimpan confusion matrix heatmap.

    Args:
        preds:      predicted class indices (N,)
        labels:     ground truth (N,)
        class_names: list nama kelas
        save_path:  path untuk menyimpan plot
        normalize:  normalisasi per baris (recall)
    """
    names = class_names or EMOTION_CLASSES
    cm    = confusion_matrix(labels, preds)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")


def plot_coverage_vs_risk(
    coverages: List[float],
    risks: List[float],
    save_path: str = "results/plots/coverage_vs_risk.png",
):
    """
    Plot Coverage vs Selective Risk curve.

    Args:
        coverages: list coverage values
        risks:     list selective risk values
        save_path: path untuk menyimpan plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, risks, marker="o", linewidth=2, color="steelblue", markersize=5)
    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Risk (Error Rate)", fontsize=12)
    ax.set_title("Coverage vs Selective Risk", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Coverage-Risk curve saved to {save_path}")


def compute_fps(total_frames: int, elapsed_seconds: float) -> float:
    """Menghitung frames per second."""
    if elapsed_seconds <= 0:
        return 0.0
    return total_frames / elapsed_seconds
