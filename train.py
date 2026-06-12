"""
train.py
Entry point untuk training FER model.

Usage:
    python train.py --config configs/fer2013.yaml
    python train.py --config configs/rafdb.yaml --csv datasets/fer2013/fer2013.csv
"""
import argparse
import sys
import torch

sys.path.insert(0, ".")

from src.fer.model import FERModel
from src.fer.dataset import FERDataset, get_dataloader
from src.fer.transforms import get_transforms
from src.fer.trainer import Trainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train FER Model")
    parser.add_argument("--config", type=str, default="configs/fer2013.yaml", help="Path ke config file")
    parser.add_argument("--csv",    type=str, default=None, help="Path ke fer2013.csv (untuk format CSV)")
    parser.add_argument("--resume", type=str, default=None, help="Path ke checkpoint untuk melanjutkan training")
    parser.add_argument("--device", type=str, default=None, help="cuda atau cpu (default: auto)")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = get_logger("train", log_file="checkpoints/logs/train.log")

    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {cfg.dataset.name} | Backbone: {cfg.model.backbone}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device.upper()}")

    # ── Transforms ──────────────────────────────────────────────────────
    train_tf = get_transforms("train", image_size=cfg.dataset.image_size)
    val_tf   = get_transforms("val",   image_size=cfg.dataset.image_size)

    # ── Dataset ─────────────────────────────────────────────────────────
    train_ds = FERDataset(
        root=cfg.dataset.root, split="train",
        transform=train_tf, csv_file=args.csv
    )
    val_ds = FERDataset(
        root=cfg.dataset.root, split="val",
        transform=val_tf, csv_file=args.csv
    )

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = get_dataloader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        use_weighted_sampler=True,   # atasi class imbalance
    )
    val_loader = get_dataloader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────────
    model = FERModel(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        backbone=cfg.model.backbone,
    )

    # Resume dari checkpoint
    if args.resume:
        import torch as _t
        ckpt = _t.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from: {args.resume}")

    # Freeze backbone di awal (opsional: uncomment untuk feature extraction phase)
    # model.freeze_backbone()

    logger.info(f"Trainable params: {model.count_parameters():,}")

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        label_smoothing=cfg.training.label_smoothing,
        use_amp=cfg.training.use_amp,
        patience=cfg.training.early_stopping_patience,
        checkpoint_dir=cfg.paths.checkpoints,
        log_dir=cfg.paths.logs,
        device=device,
        warmup_epochs=cfg.training.get("warmup_epochs", 0),
        backbone_lr=cfg.training.get("backbone_lr", None),
        backbone_lr_factor=cfg.training.get("backbone_lr_factor", 0.1),
        freeze_backbone_epochs=cfg.training.get("freeze_backbone_epochs", 0),
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        use_tensorboard=cfg.training.get("use_tensorboard", False),
        min_lr=cfg.training.get("min_lr", 1e-6),
    )

    history = trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
