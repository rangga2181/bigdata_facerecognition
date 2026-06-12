"""
src/fer/trainer.py
Training loop dengan Focal Loss, MixUp, early stopping, checkpointing,
dan tracking berdasarkan macro F1-score (lebih adil untuk imbalanced dataset).

CPU-Optimized:
  - AMP hanya aktif jika CUDA tersedia (tidak di CPU)
  - num_workers=0 untuk Windows CPU
"""
import os
import time
import csv
import math
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from .model import FERModel

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class EarlyStopping:
    """
    Early stopping berdasarkan macro F1 pada validation set.
    Berhenti jika F1 tidak membaik selama 'patience' epoch.
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience    = patience
        self.delta       = delta
        self.counter     = 0
        self.best_score  = float("-inf")
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Training manager untuk FER model.
    Mendukung Focal Loss, MixUp, class weights, dan early stopping by F1.

    Args:
        model:                   FERModel instance
        train_loader:            DataLoader untuk training set
        val_loader:              DataLoader untuk validation set
        epochs:                  jumlah epoch maksimum
        lr:                      learning rate awal (head)
        weight_decay:            L2 regularization
        label_smoothing:         label smoothing epsilon
        use_amp:                 mixed precision (diabaikan di CPU)
        patience:                early stopping patience
        checkpoint_dir:          direktori simpan model
        log_dir:                 direktori log CSV
        device:                  'cuda' atau 'cpu'
        warmup_epochs:           epoch warmup LR
        backbone_lr_factor:      faktor LR backbone vs head
        freeze_backbone_epochs:  epoch awal dengan backbone frozen
        gradient_accumulation_steps: akumulasi gradient
        use_tensorboard:         log ke TensorBoard
        min_lr:                  minimum learning rate
        use_focal_loss:          gunakan Focal Loss
        focal_gamma:             gamma untuk Focal Loss
        use_mixup:               aktifkan MixUp augmentation
        mixup_alpha:             alpha untuk distribusi Beta (MixUp)
        class_weights:           tensor class weights (opsional)
    """

    def __init__(
        self,
        model: FERModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        use_amp: bool = False,            # False default untuk CPU safety
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "checkpoints/logs",
        device: Optional[str] = None,
        max_runtime_seconds: Optional[float] = None,
        warmup_epochs: int = 0,
        backbone_lr: Optional[float] = None,
        backbone_lr_factor: float = 0.1,
        freeze_backbone_epochs: int = 0,
        gradient_accumulation_steps: int = 1,
        use_tensorboard: bool = False,
        min_lr: float = 1e-6,
        monitor_metric: str = "acc",
        # New: loss and augmentation
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.epochs        = epochs
        self.max_runtime_seconds = max_runtime_seconds
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir        = Path(log_dir)
        self.warmup_epochs  = int(warmup_epochs or 0)
        self.backbone_lr    = backbone_lr
        self.backbone_lr_factor = backbone_lr_factor
        self.freeze_backbone_epochs = int(freeze_backbone_epochs or 0)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps or 1)
        self.use_tensorboard = bool(use_tensorboard)
        self.min_lr = float(min_lr)
        self.use_mixup   = use_mixup
        self.mixup_alpha = mixup_alpha

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Device — AMP hanya aktif di CUDA
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and (self.device == "cuda")
        self.model.to(self.device)

        # ── Loss Function ──────────────────────────────────────────────────
        if use_focal_loss:
            from .focal_loss import FocalLoss
            alpha = class_weights.to(self.device) if class_weights is not None else None
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=alpha,
                label_smoothing=label_smoothing,
            )
            print(f"[Trainer] Loss: FocalLoss(gamma={focal_gamma}, label_smoothing={label_smoothing})")
        else:
            weight = class_weights.to(self.device) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=label_smoothing,
            )
            print(f"[Trainer] Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")

        # ── Optimizer dengan Differential LR (head vs backbone) ───────────
        head_params     = list(self.model.classifier.parameters()) if hasattr(self.model, "classifier") else []
        backbone_params = list(self.model.backbone.parameters())   if hasattr(self.model, "backbone") else []

        param_groups: List[dict] = []
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})
        if backbone_params:
            b_lr = float(self.backbone_lr) if self.backbone_lr is not None else lr * float(self.backbone_lr_factor)
            param_groups.append({"params": backbone_params, "lr": b_lr})

        if param_groups:
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # ── Scheduler: warmup + cosine ─────────────────────────────────────
        if self.warmup_epochs > 0 and self.warmup_epochs < epochs:
            base_lrs = [g["lr"] for g in self.optimizer.param_groups]

            def make_lr_lambda(base_lr: float, min_lr: float, warmup: int, total: int):
                min_factor = float(min_lr) / float(base_lr)

                def lr_fn(epoch: int):
                    if epoch < warmup:
                        return float(epoch + 1) / float(warmup)
                    else:
                        progress = float(epoch - warmup) / max(1.0, float(total - warmup))
                        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                        return cos * (1.0 - min_factor) + min_factor
                return lr_fn

            lr_lambdas = [make_lr_lambda(bl, self.min_lr, self.warmup_epochs, epochs) for bl in base_lrs]
            self.scheduler = LambdaLR(self.optimizer, lr_lambdas)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=self.min_lr)

        # ── AMP Scaler (hanya CUDA) ────────────────────────────────────────
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── Early Stopping ─────────────────────────────────────────────────
        self.monitor_metric = monitor_metric  # "acc" or "f1"
        self.early_stopping = EarlyStopping(patience=patience)

        # ── TensorBoard ───────────────────────────────────────────────────
        self.writer = None
        if self.use_tensorboard and _TB_AVAILABLE:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except Exception:
                self.writer = None

        # ── Log CSV ───────────────────────────────────────────────────────
        self.log_path = self.log_dir / "training_log.csv"
        self._init_log()

    def _init_log(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "time_s"])

    def _write_log(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, elapsed):
        current_lr = self.optimizer.param_groups[0]["lr"]
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}", f"{train_acc:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}",
                f"{current_lr:.6f}", f"{elapsed:.1f}",
            ])
        if self.writer is not None:
            self.writer.add_scalar("train/loss", float(train_loss), epoch)
            self.writer.add_scalar("train/acc",  float(train_acc),  epoch)
            self.writer.add_scalar("val/loss",   float(val_loss),   epoch)
            self.writer.add_scalar("val/acc",    float(val_acc),    epoch)
            self.writer.add_scalar("val/f1",     float(val_f1),     epoch)
            self.writer.add_scalar("train/lr",   float(current_lr), epoch)

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        """Menjalankan satu epoch (training atau validation)."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct    = 0
        total      = 0
        all_preds  = []
        all_labels = []

        ctx  = torch.enable_grad() if training else torch.no_grad()
        desc = "Train" if training else "Val  "

        with ctx:
            pbar = tqdm(loader, desc=desc, leave=False, ncols=90)
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if training and (batch_idx % self.gradient_accumulation_steps == 0):
                    self.optimizer.zero_grad(set_to_none=True)

                # MixUp (hanya saat training)
                use_mixup_this_batch = training and self.use_mixup and self.mixup_alpha > 0
                if use_mixup_this_batch:
                    from .mixup import mixup_data, mixup_criterion
                    mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=self.mixup_alpha)
                    input_images = mixed_images
                else:
                    input_images = images

                if self.use_amp:
                    with autocast(self.device, enabled=True):
                        logits = self.model(input_images)
                        if use_mixup_this_batch:
                            from .mixup import mixup_criterion
                            loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                        else:
                            loss = self.criterion(logits, labels)
                else:
                    logits = self.model(input_images)
                    if use_mixup_this_batch:
                        from .mixup import mixup_criterion
                        loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                    else:
                        loss = self.criterion(logits, labels)

                if training:
                    scaled_loss = loss / float(self.gradient_accumulation_steps)
                    if self.use_amp:
                        self.scaler.scale(scaled_loss).backward()
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    else:
                        scaled_loss.backward()
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()

                preds = logits.argmax(dim=-1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)
                total_loss += loss.item() * labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = total_loss / total
        avg_acc  = correct / total

        # Hitung macro F1 (lebih representatif untuk imbalanced dataset)
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        return {"loss": avg_loss, "acc": avg_acc, "f1": macro_f1}

    def save_checkpoint(self, epoch: int, val_acc: float, val_f1: float, is_best: bool = False):
        """Menyimpan checkpoint model."""
        state = {
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc":          val_acc,
            "val_f1":           val_f1,
            "config": {
                "num_classes": self.model.num_classes,
                "backbone":    "efficientnet_b0",
                "dropout":     0.4,
            },
        }
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pth"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"  ✅ Best model saved (epoch={epoch}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f})")

    def train(self) -> Dict[str, Any]:
        """
        Menjalankan training loop lengkap.
        Best model dipilih berdasarkan metric yang di-monitor (accuracy atau macro F1).

        Returns:
            Dict berisi history training
        """
        print(f"🚀 Training on {self.device.upper()}")

        if self.freeze_backbone_epochs > 0:
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()

        print(f"   Trainable params : {self.model.count_parameters():,}")
        print(f"   AMP              : {self.use_amp}")
        print(f"   MixUp            : {self.use_mixup} (alpha={self.mixup_alpha})")
        print(f"   Monitor Metric   : {self.monitor_metric}")
        print()

        best_val_metric = 0.0
        history = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss":   [], "val_acc":   [], "val_f1":   [],
        }
        train_started_at = time.time()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics   = self._run_epoch(self.val_loader,   training=False)

            # Scheduler step
            try:
                self.scheduler.step()
            except Exception:
                try:
                    self.scheduler.step(epoch)
                except Exception:
                    pass

            elapsed = time.time() - t0

            # Log
            self._write_log(
                epoch,
                train_metrics["loss"], train_metrics["acc"],
                val_metrics["loss"],   val_metrics["acc"],
                val_metrics["f1"],     elapsed,
            )

            for key in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]:
                split, metric = key.split("_", 1)
                history[key].append((train_metrics if split == "train" else val_metrics)[metric])

            # Print summary
            print(
                f"Epoch [{epoch:3d}/{self.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} F1: {val_metrics['f1']:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Monitor metric value
            monitor_val = val_metrics["acc"] if self.monitor_metric == "acc" else val_metrics["f1"]

            # Save best by monitor_metric
            is_best = monitor_val > best_val_metric
            if is_best:
                best_val_metric = monitor_val
            self.save_checkpoint(epoch, val_metrics["acc"], val_metrics["f1"], is_best=is_best)

            # Unfreeze backbone setelah N epoch
            if self.freeze_backbone_epochs and epoch == self.freeze_backbone_epochs:
                if hasattr(self.model, "unfreeze_backbone"):
                    self.model.unfreeze_backbone()
                    print(f"🔓 Backbone unfrozen at epoch {epoch}.")

            # Early stopping berdasarkan monitor_metric
            if self.early_stopping(monitor_val):
                print(f"\n⏹️  Early stopping triggered at epoch {epoch} (best {self.monitor_metric}={best_val_metric:.4f})")
                break

            # Runtime check
            if self.max_runtime_seconds is not None and epoch < self.epochs:
                total_elapsed = time.time() - train_started_at
                remaining     = self.max_runtime_seconds - total_elapsed
                avg_epoch_t   = total_elapsed / epoch
                if remaining <= 0 or remaining < avg_epoch_t:
                    print(f"\n⏹️  Max runtime reached at epoch {epoch}")
                    break

        print(f"\n✅ Training complete. Best val {self.monitor_metric}: {best_val_metric:.4f}")
        return history
