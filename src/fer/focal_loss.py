"""
src/fer/focal_loss.py
Focal Loss dan Class-Balanced Loss untuk imbalanced FER.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance.
    Menurunkan bobot sampel yang mudah (confidence tinggi) sehingga
    model lebih fokus pada sampel yang sulit dan kelas minoritas.

    Args:
        gamma:          focusing parameter (default 2.0)
        alpha:          class weights tensor (optional, shape [num_classes])
        label_smoothing: label smoothing epsilon
        reduction:      'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha          # class weights
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits dari model
            targets: (B,)   ground-truth integer labels

        Returns:
            scalar loss
        """
        num_classes = logits.size(1)

        # Label smoothing
        if self.label_smoothing > 0.0:
            # soft targets
            with torch.no_grad():
                smooth_val   = self.label_smoothing / num_classes
                one_hot      = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
                soft_targets = one_hot * (1.0 - self.label_smoothing) + smooth_val
        else:
            soft_targets = None

        # Log-softmax probabilities
        log_prob = F.log_softmax(logits, dim=-1)     # (B, C)
        prob     = log_prob.exp()                     # (B, C)

        # Gather p_t (probability of true class)
        prob_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - prob_t) ** self.gamma

        # Cross entropy (with optional label smoothing)
        if soft_targets is not None:
            ce = -(soft_targets * log_prob).sum(dim=1)  # (B,)
        else:
            ce = F.nll_loss(log_prob, targets, reduction="none")  # (B,)

        # Apply focal weight
        loss = focal_weight * ce  # (B,)

        # Apply class alpha weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]             # (B,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_class_weights(dataset, num_classes: int = 8) -> torch.Tensor:
    """
    Hitung class weights berdasarkan inverse frequency.
    Kelas lebih jarang → bobot lebih tinggi.

    Args:
        dataset: FERDataset instance (punya .samples list of (path, label))
        num_classes: jumlah kelas

    Returns:
        Tensor (num_classes,) berisi normalized class weights
    """
    labels   = [s[1] for s in dataset.samples]
    counts   = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1

    # Inverse frequency, handle zero-count kelas
    weights  = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    weights *= num_classes   # scale agar mean weight = 1.0
    return weights


def build_loss(
    dataset=None,
    num_classes: int = 8,
    use_focal: bool = True,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    use_class_weights: bool = True,
) -> nn.Module:
    """
    Factory function untuk membangun loss function.

    Args:
        dataset:           FERDataset (dipakai untuk hitung class weights)
        num_classes:       jumlah kelas
        use_focal:         gunakan Focal Loss, jika False pakai CrossEntropy
        gamma:             Focal Loss gamma
        label_smoothing:   label smoothing epsilon
        use_class_weights: terapkan class weights

    Returns:
        nn.Module loss function
    """
    alpha = None
    if use_class_weights and dataset is not None:
        alpha = get_class_weights(dataset, num_classes)
        print(f"[Loss] Class weights: {alpha.numpy().round(3)}")

    if use_focal:
        print(f"[Loss] Using FocalLoss(gamma={gamma}, label_smoothing={label_smoothing})")
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    else:
        print(f"[Loss] Using CrossEntropyLoss(label_smoothing={label_smoothing})")
        weight = alpha
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
