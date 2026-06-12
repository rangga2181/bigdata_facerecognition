"""
src/fer/mixup.py
MixUp augmentation on-the-fly di dalam training loop.
CPU-friendly, tidak ada operasi berat.
"""
import numpy as np
import torch
from typing import Tuple


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Terapkan MixUp augmentation ke satu batch.

    x_mixed = lam * x[i] + (1 - lam) * x[perm[i]]
    Saat training, loss dihitung dengan dua target:
        loss = lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b)

    Args:
        x:     input tensor (B, C, H, W)
        y:     label tensor (B,)
        alpha: Beta distribution parameter (0 = no mixup, >0 = active)

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam   = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm  = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y_a = y
    y_b = y[perm]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Hitung MixUp loss.

    Args:
        criterion: loss function (misal FocalLoss)
        pred:  model output logits (B, C)
        y_a:   original labels (B,)
        y_b:   shuffled labels (B,)
        lam:   mixing coefficient

    Returns:
        scalar mixed loss
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
