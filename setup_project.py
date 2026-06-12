"""
setup_project.py
Membuat seluruh struktur direktori proyek secara otomatis.

Usage:
    python setup_project.py
"""
import os
from pathlib import Path

# ─── Definisi struktur direktori ─────────────────────────────────────────────

DIRS = [
    # ── Data ──────────────────────────────────────────────
    "data/raw/fer2013",
    "data/raw/rafdb",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/augmented",
    "data/cache",

    # ── Notebooks ─────────────────────────────────────────
    "notebooks",

    # ── Source code ───────────────────────────────────────
    "src/detection",
    "src/alignment",
    "src/quality",
    "src/fer",
    "src/calibration",
    "src/rejection",
    "src/realtime",
    "src/utils",

    # ── Configs ───────────────────────────────────────────
    "configs",

    # ── Checkpoints ───────────────────────────────────────
    "checkpoints/logs",

    # ── Results ───────────────────────────────────────────
    "results/plots/eda",
    "results/plots/training",
    "results/plots/calibration",
    "results/plots/evaluation",
    "results/reports",
    "results/screenshots",
    "results/recordings",

    # ── Assets ────────────────────────────────────────────
    "assets/sample_faces",
    "assets/demo_images",
]

# ─── Placeholder .gitkeep files ──────────────────────────────────────────────

GITKEEP_DIRS = [
    "data/raw/fer2013",
    "data/raw/rafdb",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/augmented",
    "checkpoints",
    "results/plots/eda",
    "results/plots/training",
    "results/plots/calibration",
    "results/plots/evaluation",
    "results/reports",
    "results/screenshots",
    "assets/sample_faces",
    "assets/demo_images",
]


def create_structure():
    root = Path(".")
    created = 0

    print("📁 Creating project directory structure...\n")

    for d in DIRS:
        path = root / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {d}/")
            created += 1
        else:
            print(f"  ⏭️  Exists:  {d}/")

    # .gitkeep files
    for d in GITKEEP_DIRS:
        gk = root / d / ".gitkeep"
        if not gk.exists():
            gk.touch()

    print(f"\n✅ Done! Created {created} new directories.")
    print("\n📋 Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. Place FER2013 CSV at: data/raw/fer2013/fer2013.csv")
    print("     OR extract folders to: data/raw/fer2013/{train,val,test}/")
    print("  3. python notebooks/convert_notebooks.py")
    print("  4. jupyter notebook  (then open notebooks/)")
    print("  5. python train.py --config configs/fer2013.yaml")


if __name__ == "__main__":
    create_structure()
