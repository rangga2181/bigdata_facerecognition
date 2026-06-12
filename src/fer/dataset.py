"""
src/fer/dataset.py
Dataset loader untuk FER2013 dan RAF-DB.
"""
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


EMOTION_CLASSES = {
    0: "angry",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "suprise",
}


class FERDataset(Dataset):
    """
    Dataset generik untuk FER.
    Mendukung struktur folder:
        root/
          train/
            Angry/  img1.jpg ...
            Happy/  img1.jpg ...
            ...
          val/
            ...
          test/
            ...

    Atau untuk FER2013 CSV format:
        root/fer2013.csv

    Args:
        root:      path ke direktori dataset
        split:     'train', 'val', atau 'test'
        transform: transform pipeline
        csv_file:  path ke fer2013.csv jika menggunakan format CSV
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        csv_file: Optional[str] = None,
    ):
        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.samples   = []  # list of (pil_image_or_path, label)

        if csv_file:
            self._load_fer2013_csv(csv_file, split)
        else:
            self._load_from_folder(split)

    def _load_fer2013_csv(self, csv_path: str, split: str):
        """Load FER2013 dari file CSV resmi."""
        split_map = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
        target_usage = split_map.get(split, "Training")

        df = pd.read_csv(csv_path)
        subset = df[df["Usage"] == target_usage]

        for _, row in subset.iterrows():
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(pixels).convert("RGB")
            label = int(row["emotion"])
            self.samples.append((img, label))

    def _load_from_folder(self, split: str):
        """Load dataset dari struktur folder ImageFolder."""
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        # Map class name → label index, case-insensitive
        name_to_idx = {v.lower(): k for k, v in EMOTION_CLASSES.items()}

        for cls_name in class_names:
            label = name_to_idx.get(cls_name.lower())
            if label is None:
                # Fallback: urutan alfabet
                label = class_names.index(cls_name)

            cls_dir = split_dir / cls_name
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item, label = self.samples[idx]

        # Load image
        if isinstance(item, str):
            image = Image.open(item).convert("RGB")
        else:
            image = item  # sudah PIL image (dari CSV)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Menghitung class weights untuk mengatasi class imbalance."""
        labels = [s[1] for s in self.samples]
        num_classes = len(EMOTION_CLASSES)
        class_counts = torch.zeros(num_classes)
        for lbl in labels:
            class_counts[lbl] += 1
        weights = 1.0 / (class_counts + 1e-6)
        weights /= weights.sum()
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """Menghitung sample weights untuk WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor([class_weights[s[1]] for s in self.samples])
        return sample_weights

    @property
    def classes(self) -> list:
        """Mengembalikan daftar nama kelas emosi."""
        return list(EMOTION_CLASSES.values())



def get_dataloader(
    dataset: FERDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    Membuat DataLoader dari FERDataset.

    Args:
        dataset: FERDataset instance
        batch_size: ukuran batch
        shuffle: apakah di-shuffle (diabaikan jika use_weighted_sampler=True)
        num_workers: jumlah worker untuk loading
        use_weighted_sampler: apakah menggunakan WeightedRandomSampler

    Returns:
        DataLoader
    """
    sampler = None
    if use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # sampler dan shuffle tidak bisa bersamaan

    # Default pin_memory to True only when CUDA is available to avoid
    # the DataLoader warning when running on CPU-only environments.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(shuffle or sampler is not None),
    )
