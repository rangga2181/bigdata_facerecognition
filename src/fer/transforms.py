"""
src/fer/transforms.py
Augmentation transforms untuk training dan validasi.
CPU-Optimized: augmentasi lebih agresif tanpa operasi berat GPU.
"""
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_transforms(split: str = "train", image_size: int = 224):
    """
    Mengembalikan transform pipeline untuk split yang ditentukan.

    CPU-optimized:
      - Training: augmentasi lebih kuat untuk generalisasi lebih baik
      - Val/Test : hanya resize + normalize (tidak ada augmentasi)

    Args:
        split:      'train', 'val', atau 'test'
        image_size: ukuran output gambar

    Returns:
        torchvision transform pipeline
    """
    # ImageNet normalization stats (EfficientNet pretrained)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            # Step 1: Resize sedikit lebih besar lalu crop → simulasi pan/zoom
            T.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BILINEAR),
            T.RandomCrop(image_size),

            # Step 2: Geometric augmentasi
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),           # lebih besar dari 15
            T.RandomPerspective(distortion_scale=0.2, p=0.3),  # simulasi pose variation
            T.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),             # pan ringan
                shear=10,                           # shear untuk variasi wajah miring
            ),

            # Step 3: Color augmentasi lebih kuat
            T.ColorJitter(
                brightness=0.5,   # dari 0.3
                contrast=0.5,     # dari 0.3
                saturation=0.3,   # dari 0.2
                hue=0.08,         # dari 0.05
            ),
            T.RandomGrayscale(p=0.15),              # dari 0.1

            # Step 4: Blur (simulasi foto blur / kualitas rendah)
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

            # Step 5: Normalization
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),

            # Step 6: Random Erasing (simulasi oklusi wajah)
            T.RandomErasing(p=0.4, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
        ])
    else:
        # val / test: NO augmentation — hanya resize dan normalize
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def get_inference_transform(image_size: int = 224):
    """
    Transform untuk inference real-time (numpy BGR → tensor).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(image_size: int = 224):
    """
    Test-Time Augmentation (TTA) transforms.
    Mengembalikan list 5 transform berbeda yang akan di-ensemble.
    Cocok untuk inference final, sedikit meningkatkan accuracy.

    Returns:
        List[transform]
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base = [
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
    ]
    normalize = [T.ToTensor(), T.Normalize(mean=mean, std=std)]

    return [
        T.Compose(base + normalize),                                         # original
        T.Compose(base + [T.RandomHorizontalFlip(p=1.0)] + normalize),      # hflip
        T.Compose([
            T.Resize((image_size + 16, image_size + 16)),
            T.CenterCrop(image_size),
        ] + normalize),                                                       # center crop
        T.Compose(base + [T.ColorJitter(brightness=0.2, contrast=0.2)] + normalize),  # brighter
        T.Compose([
            T.Resize((image_size + 16, image_size + 16)),
            T.RandomCrop(image_size),
        ] + normalize),                                                       # random crop
    ]
