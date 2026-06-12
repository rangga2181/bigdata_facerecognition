"""
src/calibration/calibration_evaluator.py
Mengevaluasi kalibrasi model: ECE, Brier Score, reliability diagram.
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
from typing import Tuple, Dict
from pathlib import Path


class CalibrationEvaluator:
    """
    Evaluasi kalibrasi probabilitas prediksi model.

    Menyediakan:
        - Expected Calibration Error (ECE)
        - Brier Score
        - Reliability Diagram (plot)
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins

    def compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Menghitung Expected Calibration Error (ECE).

        Args:
            probs:  numpy array (N, C) softmax probabilities
            labels: numpy array (N,) ground truth class indices

        Returns:
            ECE score (0 = perfectly calibrated)
        """
        max_probs = probs.max(axis=1)
        preds     = probs.argmax(axis=1)
        correct   = (preds == labels).astype(float)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask   = (max_probs >= lo) & (max_probs < hi)
            if mask.sum() == 0:
                continue
            bin_acc  = correct[mask].mean()
            bin_conf = max_probs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)

        return float(ece / len(labels))

    def compute_brier_score(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Menghitung Brier Score.

        Args:
            probs:  numpy array (N, C)
            labels: numpy array (N,)

        Returns:
            Brier Score (0 = perfect)
        """
        n_classes = probs.shape[1]
        one_hot   = np.eye(n_classes)[labels]
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    def compute_all(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Menghitung semua calibration metrics sekaligus."""
        ece    = self.compute_ece(probs, labels)
        brier  = self.compute_brier_score(probs, labels)
        return {"ECE": ece, "BrierScore": brier}

    def plot_reliability_diagram(
        self,
        probs_before: np.ndarray,
        probs_after: np.ndarray,
        labels: np.ndarray,
        save_path: str = "results/plots/calibration_curve.png",
    ):
        """
        Membuat reliability diagram before & after calibration.

        Args:
            probs_before: probabilities sebelum kalibrasi (N, C)
            probs_after:  probabilities setelah kalibrasi (N, C)
            labels:       ground truth (N,)
            save_path:    path untuk menyimpan plot
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, probs, title in [
            (axes[0], probs_before, "Before Calibration"),
            (axes[1], probs_after,  "After Calibration"),
        ]:
            max_probs = probs.max(axis=1)
            preds     = probs.argmax(axis=1)
            correct   = (preds == labels).astype(float)

            accs  = []
            confs = []
            for i in range(self.n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask   = (max_probs >= lo) & (max_probs < hi)
                if mask.sum() == 0:
                    accs.append(0.0)
                    confs.append((lo + hi) / 2)
                else:
                    accs.append(correct[mask].mean())
                    confs.append(max_probs[mask].mean())

            ece = self.compute_ece(probs, labels)

            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.bar(bin_centers, accs, width=1.0 / self.n_bins, alpha=0.7, color="steelblue", label="Model accuracy")
            ax.step(bin_centers, confs, where="mid", color="tomato", linewidth=2, label="Mean confidence")
            ax.set_title(f"{title}\nECE = {ece:.4f}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Reliability diagram saved to {save_path}")
