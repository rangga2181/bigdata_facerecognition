# %% [markdown]
# # 🌡️ Notebook 05 — Confidence Calibration
# ### Quality-Aware FER Project | Temperature Scaling
# **Dataset:** `dataset/validation`  |  **Kelas:** 8 emosi

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
from pathlib import Path

# Temukan ROOT project secara robust (baik jika dijalankan dari folder 'notebooks' maupun root project)
current_path = Path(".").resolve()
if (current_path / "src").exists() and (current_path / "configs").exists():
    ROOT = current_path
elif (current_path.parent / "src").exists() and (current_path.parent / "configs").exists():
    ROOT = current_path.parent
else:
    ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
plt.style.use("dark_background")
SAVE_DIR = ROOT / "results" / "plots" / "calibration"; SAVE_DIR.mkdir(parents=True, exist_ok=True)
device   = "cuda" if torch.cuda.is_available() else "cpu"

# Helper: tampilkan gambar yang disimpan ke disk secara inline
def inline_show(path, figsize=(14, 5), title=None):
    """Baca file gambar yang disimpan dan tampilkan inline di Jupyter/IPython."""
    import matplotlib.image as mpimg
    img = mpimg.imread(str(path))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=12)
    plt.tight_layout()
    plt.show()

print(f"✅ Device: {device.upper()}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
from src.fer.model import FERModel
from src.fer.dataset import FERDataset, get_dataloader
from src.fer.transforms import get_transforms
from src.utils.config_loader import load_config

cfg       = load_config(str(ROOT / "configs" / "default.yaml"))
ckpt_path = ROOT / "checkpoints" / "best_model.pth"

if not ckpt_path.exists():
    print(f"❌ Checkpoint not found: {ckpt_path}")
    print("   Jalankan Notebook 04 (Training) terlebih dahulu.")
    raise FileNotFoundError(str(ckpt_path))

model = FERModel.load_from_checkpoint(str(ckpt_path), device=device)
model.eval()
print(f"✅ Model loaded: {ckpt_path}")

# %% [markdown]
# ## 2. Collect Logits & Labels dari Validation Set

# %%
val_tf     = get_transforms("val", image_size=cfg.dataset.image_size)
val_ds     = FERDataset(root=str(ROOT/"dataset"), split="validation", transform=val_tf)
val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=0)
print(f"Val samples: {len(val_ds)}")

all_logits, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0).numpy()

probs_before = torch.softmax(all_logits, dim=-1).numpy()
print(f"Collected {len(all_labels)} val samples.")

# %% [markdown]
# ## 3. Analisis Over-Confidence (Before Calibration)

# %%
max_confs_before = probs_before.max(axis=1)
preds_before     = probs_before.argmax(axis=1)
correct_before   = (preds_before == all_labels).astype(float)

print(f"Before Calibration:")
print(f"  Mean Max Confidence : {max_confs_before.mean():.4f}")
print(f"  Accuracy            : {correct_before.mean():.4f}")
print(f"  Avg Confidence - Acc: {max_confs_before.mean() - correct_before.mean():.4f}  ← over-confidence gap")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Before Calibration — Confidence Analysis", fontsize=13, color="white")

# Confidence histogram
ax = axes[0]
ax.hist(max_confs_before, bins=50, color="#3b82f6", alpha=0.8, edgecolor="none")
ax.axvline(max_confs_before.mean(), color="#f97316", linewidth=2, linestyle="--",
           label=f"Mean conf={max_confs_before.mean():.3f}")
ax.axvline(correct_before.mean(), color="#22c55e", linewidth=2, linestyle="--",
           label=f"Accuracy={correct_before.mean():.3f}")
ax.set_title("Max Confidence Distribution", color="white"); ax.set_xlabel("Confidence", color="white")
ax.set_ylabel("Count", color="white"); ax.legend(facecolor="#1e1e2e", labelcolor="white")
ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Calibration bin analysis
ax2 = axes[1]
n_bins    = 15
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_accs  = []; bin_confs = []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (max_confs_before >= lo) & (max_confs_before < hi)
    if mask.sum() > 0:
        bin_accs.append(correct_before[mask].mean())
        bin_confs.append(max_confs_before[mask].mean())
    else:
        bin_accs.append(0); bin_confs.append((lo+hi)/2)

bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
ax2.plot([0,1],[0,1],"k--",lw=2,label="Perfect calibration")
ax2.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.7, color="#3b82f6", label="Accuracy")
ax2.step(bin_centers, bin_confs, where="mid", color="#f97316", lw=2, label="Confidence")
ax2.set_title("Reliability Diagram (Before)", color="white"); ax2.set_xlabel("Confidence")
ax2.set_ylabel("Accuracy"); ax2.legend(facecolor="#1e1e2e", labelcolor="white")
ax2.set_facecolor("#1e1e2e"); ax2.tick_params(colors="white"); ax2.set_xlim(0,1); ax2.set_ylim(0,1)

plt.tight_layout()
plt.savefig(SAVE_DIR/"01_before_calibration.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# %% [markdown]
# ## 4. Temperature Scaling Calibration

# %%
from src.calibration.temperature_scaling import TemperatureScaling

ts = TemperatureScaling(temperature=1.0)
T_optimal = ts.calibrate(
    model=model,
    val_loader=val_loader,
    device=device,
    max_iter=50,
    lr=0.01,
)
ts.save(str(ROOT / "checkpoints" / "temperature.pth"))
print(f"\n🌡️  Optimal Temperature T = {T_optimal:.4f}")
print(f"   T > 1 → model was over-confident (softens probabilities)")

# %% [markdown]
# ## 5. After Calibration — Evaluation

# %%
probs_after = ts.predict_proba(all_logits).numpy()
max_confs_after = probs_after.max(axis=1)
preds_after     = probs_after.argmax(axis=1)
correct_after   = (preds_after == all_labels).astype(float)

print(f"After Calibration (T={T_optimal:.3f}):")
print(f"  Mean Max Confidence : {max_confs_after.mean():.4f}")
print(f"  Accuracy            : {correct_after.mean():.4f}")
print(f"  Over-confidence gap : {max_confs_after.mean() - correct_after.mean():.4f}")

# %% [markdown]
# ## 6. Reliability Diagram — Before vs After

# %%
from src.calibration.calibration_evaluator import CalibrationEvaluator

cal_eval = CalibrationEvaluator(n_bins=15)
ece_before = cal_eval.compute_ece(probs_before, all_labels)
ece_after  = cal_eval.compute_ece(probs_after,  all_labels)
brier_bef  = cal_eval.compute_brier_score(probs_before, all_labels)
brier_aft  = cal_eval.compute_brier_score(probs_after,  all_labels)

print(f"\n📊 Calibration Metrics:")
print(f"{'Metric':<20} {'Before':>10} {'After':>10} {'Δ':>10}")
print("-"*52)
print(f"{'ECE':<20} {ece_before:>10.4f} {ece_after:>10.4f} {ece_after-ece_before:>+10.4f}")
print(f"{'Brier Score':<20} {brier_bef:>10.4f} {brier_aft:>10.4f} {brier_aft-brier_bef:>+10.4f}")

cal_eval.plot_reliability_diagram(
    probs_before, probs_after, all_labels,
    save_path=str(SAVE_DIR/"02_reliability_diagram.png")
)
inline_show(SAVE_DIR/"02_reliability_diagram.png", figsize=(14, 6), title="Reliability Diagram — Before vs After Calibration")

# %% [markdown]
# ## 7. Temperature Sensitivity Analysis

# %%
temperatures = np.arange(0.5, 3.1, 0.1)
ece_vals = []
for T in temperatures:
    scaled_probs = torch.softmax(all_logits / T, dim=-1).numpy()
    ece_vals.append(cal_eval.compute_ece(scaled_probs, all_labels))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(temperatures, ece_vals, color="#3b82f6", linewidth=2, marker="o", markersize=4)
ax.axvline(T_optimal, color="#22c55e", linewidth=2, linestyle="--", label=f"Optimal T={T_optimal:.3f}")
ax.axvline(1.0, color="#f97316", linewidth=1.5, linestyle=":", label="T=1.0 (no scaling)")
ax.set_title("ECE vs Temperature T", color="white", fontsize=13)
ax.set_xlabel("Temperature T", color="white"); ax.set_ylabel("ECE (lower = better)", color="white")
ax.legend(facecolor="#1e1e2e", labelcolor="white"); ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(SAVE_DIR/"03_temperature_sensitivity.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

print(f"\n✅ Calibration complete!")
print(f"   Temperature saved: checkpoints/temperature.pth")
print(f"   ECE: {ece_before:.4f} → {ece_after:.4f} ({'improved' if ece_after < ece_before else 'worse'})")
print("\n➡️  Next: Notebook 06 — Full Evaluation")
