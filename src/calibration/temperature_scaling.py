"""
src/calibration/temperature_scaling.py
Implementasi Temperature Scaling untuk confidence calibration.

Referensi:
    Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling wrapper untuk model klasifikasi.

    Cara kerja:
        calibrated_logits = logits / T
        calibrated_probs  = softmax(calibrated_logits)

    T > 1: menjadikan distribusi lebih smooth (kurangi overconfidence)
    T < 1: membuat distribusi lebih tajam (jarang diperlukan)

    Args:
        temperature: nilai awal temperature (default 1.0 = no scaling)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Menerapkan temperature scaling pada logits.

        Args:
            logits: raw logits (B, C)

        Returns:
            Scaled logits (B, C)
        """
        return logits / self.temperature.clamp(min=1e-6)

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = "cpu",
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Mengoptimasi nilai T menggunakan NLL loss pada validation set.

        Args:
            model:      model FER yang sudah dilatih
            val_loader: DataLoader untuk validation set
            device:     target device
            max_iter:   jumlah iterasi optimasi
            lr:         learning rate untuk optimasi T

        Returns:
            Nilai temperature optimal T
        """
        self.to(device)
        model.to(device)
        model.eval()

        # Kumpulkan semua logits dan labels dari val set
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Optimasi T
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        T = self.temperature.item()
        print(f"✅ Temperature calibrated: T = {T:.4f}")
        return T

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mengembalikan calibrated probabilities dari logits.

        Args:
            logits: raw logits (B, C)

        Returns:
            Calibrated probabilities (B, C) via softmax
        """
        with torch.no_grad():
            scaled = self.forward(logits)
            return torch.softmax(scaled, dim=-1)

    def save(self, path: str):
        """Menyimpan temperature value ke file."""
        torch.save({"temperature": self.temperature.item()}, path)
        print(f"💾 Temperature saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaling":
        """Load temperature dari file."""
        data = torch.load(path, map_location="cpu")
        instance = cls(temperature=data["temperature"])
        print(f"📂 Temperature loaded: T = {data['temperature']:.4f}")
        return instance
