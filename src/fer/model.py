"""
src/fer/model.py
FER model berbasis EfficientNet-B0 dengan classifier head yang lebih kuat.
CPU-Optimized: tetap EfficientNet-B0 agar tidak terlalu berat di CPU.
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class FERModel(nn.Module):
    """
    Facial Expression Recognition model.

    Backbone: EfficientNet-B0 (pretrained on ImageNet via timm)
    Head    : Dropout → Linear(1280→512) → BN → GELU → Dropout → Linear(512→num_classes)

    Dibanding versi sebelumnya:
      - Head lebih dalam dengan BatchNorm1d dan GELU activation
      - BatchNorm menstabilkan training dan mencegah overfitting
      - GELU (Gaussian Error Linear Unit) umumnya lebih baik dari ReLU untuk task ini

    Args:
        num_classes: jumlah kelas (default 8)
        dropout:     dropout rate pada layer pertama (default 0.4)
        pretrained:  gunakan pretrained weights (default True)
        backbone:    nama backbone timm (default 'efficientnet_b0')
    """

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.4,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load backbone dari timm (tanpa classifier head bawaan)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,   # hapus classifier head bawaan
        )
        in_features = self.backbone.num_features   # 1280 untuk efficientnet_b0

        # Custom classifier head — lebih dalam dan stabil
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),     # half dropout pada hidden layer
            nn.Linear(512, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        """Inisialisasi bobot custom head dengan Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: tensor (B, 3, H, W)

        Returns:
            logits tensor (B, num_classes)
        """
        features = self.backbone(x)       # (B, in_features)
        logits   = self.classifier(features)  # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan predicted class index."""
        return self.predict_proba(x).argmax(dim=-1)

    def freeze_backbone(self):
        """Membekukan semua layer backbone (hanya classifier yang dilatih)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Jaga BatchNorm/GroupNorm tetap di eval mode agar running stats tidak rusak
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                module.eval()
        print("[INFO] Backbone frozen. Only classifier will be trained.")

    def train(self, mode: bool = True):
        """Set training mode, while keeping backbone BN layers frozen if backbone is frozen."""
        super().train(mode)
        if mode and hasattr(self, "backbone"):
            # Cek apakah backbone sedang frozen
            is_frozen = False
            for param in self.backbone.parameters():
                is_frozen = not param.requires_grad
                break
            if is_frozen:
                for module in self.backbone.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                        module.eval()

    def unfreeze_backbone(self):
        """Membebaskan semua layer backbone untuk fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                module.train()
        print("[INFO] Backbone unfrozen. Full model fine-tuning active.")

    def count_parameters(self) -> int:
        """Menghitung jumlah parameter yang bisa dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu", **kwargs) -> "FERModel":
        """
        Load model dari checkpoint file.

        Args:
            checkpoint_path: path ke file .pth
            device:          target device
            **kwargs:        parameter model (num_classes, dropout, dll)

        Returns:
            FERModel instance dengan bobot yang di-load
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Support berbagai format checkpoint
        if "model_state_dict" in checkpoint:
            state_dict  = checkpoint["model_state_dict"]
            cfg         = checkpoint.get("config", {})
            num_classes = cfg.get("num_classes", kwargs.get("num_classes", 8))
            dropout     = cfg.get("dropout",     kwargs.get("dropout", 0.4))
            backbone    = cfg.get("backbone",    kwargs.get("backbone", "efficientnet_b0"))
        else:
            state_dict  = checkpoint
            num_classes = kwargs.get("num_classes", 8)
            dropout     = kwargs.get("dropout", 0.4)
            backbone    = kwargs.get("backbone", "efficientnet_b0")

        model = cls(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=False,
            backbone=backbone,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
