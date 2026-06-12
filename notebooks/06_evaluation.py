# %% [markdown]
# # 📈 Notebook 06 — Full Evaluation & Report
# ### Quality-Aware FER Project
#
# **Dataset:** `dataset/test` | **Kelas:** 8 emosi
#
# **Perubahan v2:**
# - **Per-Class Sample Gallery**: 1 gambar per kelas + confidence bar
#   (menampilkan model post-calibration yang sebenarnya dipakai)
# - **Baseline vs WOA Comparison**: tabel & chart perbandingan per kelas
# - **Calibrated Confidence per Class**: violin/box plot distribusi confidence

# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, json, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Temukan ROOT project
current_path = Path(".").resolve()
if (current_path / "src").exists() and (current_path / "configs").exists():
    ROOT = current_path
elif (current_path.parent / "src").exists() and (current_path.parent / "configs").exists():
    ROOT = current_path.parent
else:
    ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
plt.style.use("dark_background")
SAVE_DIR   = ROOT / "results" / "plots" / "evaluation"; SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT / "results" / "reports";              REPORT_DIR.mkdir(parents=True, exist_ok=True)
device     = "cpu"

EMOTION_CLASSES = ["angry","contempt","disgust","fear","happy","neutral","sad","suprise"]
COLORS_8        = ["#ef4444","#a16207","#a855f7","#f97316","#22c55e","#94a3b8","#3b82f6","#eab308"]
CMAP8           = dict(zip(EMOTION_CLASSES, COLORS_8))

def inline_show(path, figsize=(10, 7), title=None):
    import matplotlib.image as mpimg
    img = mpimg.imread(str(path))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img); ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=12)
    plt.tight_layout(); plt.show()

print(f"✅ Device: CPU")

# %% [markdown]
# ## 1. Load Model & Calibrator

# %%
from src.fer.model import FERModel
from src.calibration.temperature_scaling import TemperatureScaling
from src.fer.dataset import FERDataset, get_dataloader
from src.fer.transforms import get_transforms
from src.utils.config_loader import load_config

cfg       = load_config(str(ROOT / "configs" / "default.yaml"))
ckpt_path = ROOT / "checkpoints" / "best_model.pth"
temp_path = ROOT / "checkpoints" / "temperature.pth"
baseline_ckpt = ROOT / "checkpoints" / "baseline_model.pth"

if not ckpt_path.exists():
    raise FileNotFoundError(f"Model not found: {ckpt_path}. Jalankan Notebook 04 terlebih dahulu.")

model = FERModel.load_from_checkpoint(str(ckpt_path), device=device)
model.eval()
print(f"✅ Model WOA-Optimized loaded: {ckpt_path}")

calibrator = None
if temp_path.exists():
    calibrator = TemperatureScaling.load(str(temp_path))
    print(f"✅ Calibrator loaded (T={calibrator.temperature.item():.4f})")
else:
    print("⚠️  No temperature file. Using uncalibrated model.")
    print("   → Jalankan Notebook 05 untuk kalibrasi terlebih dahulu.")

# Load baseline model jika ada
model_baseline = None
if baseline_ckpt.exists():
    model_baseline = FERModel.load_from_checkpoint(str(baseline_ckpt), device=device)
    model_baseline.eval()
    print(f"✅ Baseline model loaded: {baseline_ckpt}")

# %% [markdown]
# ## 2. Inference pada Test Set

# %%
test_tf      = get_transforms("test", image_size=cfg.dataset.image_size)
test_ds      = FERDataset(root=str(ROOT/"dataset"), split="test", transform=test_tf)
test_loader  = get_dataloader(test_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Test samples: {len(test_ds):,} | Classes: {test_ds.classes}")

all_logits, all_preds_list, all_labels_list = [], [], []
latencies = []

with torch.no_grad():
    for images, labels in test_loader:
        t0     = time.perf_counter()
        logits = model(images)
        lat    = (time.perf_counter()-t0) / len(images) * 1000
        latencies.append(lat)

        if calibrator:
            probs = calibrator.predict_proba(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        preds = probs.argmax(dim=-1)
        all_logits.append(logits.cpu())
        all_preds_list.extend(preds.cpu().numpy())
        all_labels_list.extend(labels.numpy())

all_logits = torch.cat(all_logits, dim=0)
all_preds  = np.array(all_preds_list)
all_labels = np.array(all_labels_list)

if calibrator:
    all_probs = calibrator.predict_proba(all_logits).numpy()
else:
    all_probs = torch.softmax(all_logits, dim=-1).numpy()

print(f"✅ Inference selesai. Avg latency: {np.mean(latencies):.2f} ms/image")

# %% [markdown]
# ## 3. Classification Metrics

# %%
from src.utils.metrics import compute_metrics, compute_classification_report

metrics    = compute_metrics(all_preds, all_labels)
report_str = compute_classification_report(all_preds, all_labels, EMOTION_CLASSES)

print("=" * 65)
print("📊 CLASSIFICATION METRICS (WOA-Optimized + Calibrated)")
print("=" * 65)
print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
print(f"  Precision : {metrics['precision']*100:.2f}%")
print(f"  Recall    : {metrics['recall']*100:.2f}%")
print(f"  F1-Score  : {metrics['f1']*100:.2f}%")
print(f"  UAR       : {metrics['UAR']*100:.2f}%")
print(f"\n{report_str}")

# %% [markdown]
# ## 4. Confusion Matrix

# %%
from src.utils.metrics import plot_confusion_matrix
plot_confusion_matrix(all_preds, all_labels, EMOTION_CLASSES,
                      save_path=str(SAVE_DIR/"01_confusion_matrix.png"))
inline_show(SAVE_DIR/"01_confusion_matrix.png", figsize=(10, 8),
            title="Normalized Confusion Matrix (Test Set)")

# %% [markdown]
# ## 5. ⭐ Per-Class Sample Gallery
#
# Menampilkan **1 gambar per kelas** dari test set beserta:
# - True label & Predicted label
# - Confidence bar (post-calibration)
# - Warna bingkai: hijau = benar, merah = salah

# %%
# Kumpulkan path gambar dari test_ds
test_paths = []
for item, label in test_ds.samples:
    if isinstance(item, str):
        test_paths.append((item, label))
    else:
        test_paths.append((None, label))

from PIL import Image as PILImage

fig = plt.figure(figsize=(20, 10))
fig.suptitle("Per-Class Sample Gallery\n(Model WOA-Optimized + Calibrated — 1 sampel per kelas)",
             fontsize=14, color="white", fontweight="bold")

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

for cls_idx, emo in enumerate(EMOTION_CLASSES):
    row = cls_idx // 4
    col = cls_idx % 4
    ax  = fig.add_subplot(gs[row, col])

    # Cari sample dari kelas ini (ambil yang pertama dari test set)
    found_path = None
    found_true = None
    found_pred = None
    found_probs = None

    for sample_idx, (path, true_label) in enumerate(test_paths):
        if true_label == cls_idx and path is not None:
            found_path  = path
            found_true  = true_label
            found_pred  = int(all_preds[sample_idx])
            found_probs = all_probs[sample_idx]
            break

    if found_path is None:
        ax.text(0.5, 0.5, f"No sample\nfor {emo}", ha="center", va="center",
                color="gray", transform=ax.transAxes, fontsize=10)
        ax.set_facecolor("#1e1e2e"); ax.axis("off")
        continue

    # Tampilkan gambar
    try:
        img_pil = PILImage.open(found_path).convert("RGB").resize((200, 200))
        img_np  = np.array(img_pil)
        ax.imshow(img_np)
    except:
        ax.text(0.5, 0.5, "Error\nload img", ha="center", va="center",
                color="red", transform=ax.transAxes, fontsize=10)

    # Bingkai warna
    is_correct  = (found_pred == found_true)
    frame_color = "#22c55e" if is_correct else "#ef4444"
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color); spine.set_linewidth(3)

    # Confidence values
    pred_conf   = float(found_probs[found_pred]) * 100
    true_conf   = float(found_probs[found_true]) * 100
    pred_name   = EMOTION_CLASSES[found_pred]
    true_name   = EMOTION_CLASSES[found_true]

    if is_correct:
        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {pred_name} ✓\n"
            f"Conf: {true_conf:.1f}%",
            color="white",
            fontsize=8.5, fontweight="bold",
            pad=4,
        )
    else:
        ax.set_title(
            f"True: {true_name} ({true_conf:.1f}%)\n"
            f"Pred: {pred_name} ✗ ({pred_conf:.1f}%)",
            color=frame_color,
            fontsize=8.5, fontweight="bold",
            pad=4,
        )
    ax.axis("off")

    # Mini confidence bar di bawah gambar (untuk true label confidence)
    ax.annotate(
        "",
        xy=(true_conf/100, -0.05), xycoords="axes fraction",
        xytext=(0, -0.05),
        arrowprops=dict(arrowstyle="-", color=frame_color, lw=4),
    )

plt.savefig(SAVE_DIR/"02_per_class_gallery.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

print("✅ Per-class gallery disimpan.")

# %% [markdown]
# ## 6. Detail Confidence per Kelas (Calibrated)

# %%
# Untuk setiap kelas: tampilkan confidence dari model calibrated
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Confidence Distribution per Kelas (Model Calibrated)\n"
             "Hijau = benar, Merah = salah", fontsize=13, color="white")
axes = axes.flatten()

for cls_idx, emo in enumerate(EMOTION_CLASSES):
    ax     = axes[cls_idx]
    mask   = (all_labels == cls_idx)
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                color="gray", transform=ax.transAxes)
        ax.axis("off"); continue

    cls_probs_correct = all_probs[mask & (all_preds == all_labels), cls_idx]
    cls_probs_wrong   = all_probs[mask & (all_preds != all_labels), cls_idx]

    if len(cls_probs_correct):
        ax.hist(cls_probs_correct, bins=15, color="#22c55e", alpha=0.75,
                label=f"Correct (n={len(cls_probs_correct)})", edgecolor="none")
    if len(cls_probs_wrong):
        ax.hist(cls_probs_wrong, bins=15, color="#ef4444", alpha=0.75,
                label=f"Wrong   (n={len(cls_probs_wrong)})", edgecolor="none")

    # Per-class accuracy
    cls_acc = mask.sum() and (all_preds[mask] == all_labels[mask]).mean() * 100
    ax.set_title(f"{emo}\n(acc={cls_acc:.1f}%)", color=COLORS_8[cls_idx],
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("P(class|calibrated)", color="white", fontsize=8)
    ax.set_ylabel("Count", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_facecolor("#1e1e2e"); ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.5, facecolor="#1e1e2e", labelcolor="white")

plt.tight_layout()
plt.savefig(SAVE_DIR/"03_per_class_confidence.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# %% [markdown]
# ## 7. Calibration Metrics

# %%
from src.calibration.calibration_evaluator import CalibrationEvaluator

cal_eval    = CalibrationEvaluator(n_bins=15)
cal_metrics = cal_eval.compute_all(all_probs, all_labels)

print("=" * 60)
print("🌡️  CALIBRATION METRICS")
print("=" * 60)
print(f"  ECE         : {cal_metrics['ECE']:.4f}  (0 = perfect)")
print(f"  Brier Score : {cal_metrics['BrierScore']:.4f}  (0 = perfect)")
calib_label = f"T={calibrator.temperature.item():.3f}" if calibrator else "No calibration"
print(f"  Status      : {calib_label}")

# %% [markdown]
# ## 8. Baseline vs WOA — Per-Class Comparison

# %%
# Inference baseline model (jika tersedia)
if model_baseline is not None:
    all_preds_base = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits_b = model_baseline(images)
            probs_b  = torch.softmax(logits_b, dim=-1)
            preds_b  = probs_b.argmax(dim=-1)
            all_preds_base.extend(preds_b.numpy())

    all_preds_base = np.array(all_preds_base)

    # Per-class F1 comparison
    from sklearn.metrics import f1_score
    f1_per_class_base = f1_score(all_labels, all_preds_base, average=None,
                                  labels=list(range(8)), zero_division=0)
    f1_per_class_woa  = f1_score(all_labels, all_preds,      average=None,
                                  labels=list(range(8)), zero_division=0)

    # Tabel perbandingan
    print("\n" + "=" * 70)
    print("📊 PER-CLASS F1 COMPARISON: Baseline vs WOA-Optimized (Test Set)")
    print("=" * 70)
    print(f"{'Kelas':<14} {'Baseline F1':>12} {'WOA F1':>10} {'Delta':>8}  {'Status':>10}")
    print("-" * 70)
    for i, emo in enumerate(EMOTION_CLASSES):
        delta  = f1_per_class_woa[i] - f1_per_class_base[i]
        status = "↑ Naik" if delta > 0.01 else ("↓ Turun" if delta < -0.01 else "≈ Sama")
        print(f"{emo:<14} {f1_per_class_base[i]*100:>11.1f}% {f1_per_class_woa[i]*100:>9.1f}% "
              f"{delta*100:>+7.1f}%  {status:>10}")
    print("-" * 70)
    print(f"{'MACRO AVG':<14} {f1_per_class_base.mean()*100:>11.1f}% "
          f"{f1_per_class_woa.mean()*100:>9.1f}% "
          f"{(f1_per_class_woa-f1_per_class_base).mean()*100:>+7.1f}%")

    # Bar chart grouped
    x = np.arange(len(EMOTION_CLASSES)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars_b = ax.bar(x - w/2, f1_per_class_base*100, width=w,
                    color=COLORS_8, alpha=0.5, edgecolor="white", linewidth=0.4, label="Baseline")
    bars_w = ax.bar(x + w/2, f1_per_class_woa*100,  width=w,
                    color=COLORS_8, alpha=0.95, edgecolor="white", linewidth=0.4, label="WOA-Optimized")
    for i, (b, wo) in enumerate(zip(f1_per_class_base, f1_per_class_woa)):
        delta = (wo - b) * 100
        color = "#22c55e" if delta >= 0 else "#ef4444"
        ax.text(i+w/2, wo*100+0.5, f"{delta:+.1f}%", ha="center", va="bottom",
                color=color, fontsize=8, fontweight="bold")
    ax.axhline(f1_per_class_woa.mean()*100, color="yellow", linestyle="--", linewidth=2,
               label=f"WOA Macro F1={f1_per_class_woa.mean()*100:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(EMOTION_CLASSES, rotation=30, color="white")
    ax.set_title("Per-Class F1: Baseline vs WOA-Optimized (Test Set)", color="white", fontsize=13)
    ax.set_ylabel("F1 Score (%)", color="white"); ax.tick_params(axis="y", colors="white")
    ax.set_facecolor("#1e1e2e"); ax.set_ylim(0, 115)
    ax.legend(facecolor="#1e1e2e", labelcolor="white")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(SAVE_DIR/"04_per_class_f1_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()

else:
    print("⚠️  Baseline model tidak ditemukan, skip perbandingan.")

# %% [markdown]
# ## 9. Selective Prediction — Coverage vs Risk

# %%
from src.rejection.reject_option import RejectOption
from src.utils.metrics import plot_coverage_vs_risk

rejector   = RejectOption(q_threshold=0.0, c_threshold=0.5, e_threshold=2.5)
fake_quality = np.ones(len(all_probs))

coverages = []; risks = []
for c_thresh in np.arange(0.2, 0.99, 0.04):
    rejector.c_threshold = round(float(c_thresh), 2)
    cov  = rejector.compute_coverage(all_probs, fake_quality)
    risk = rejector.compute_selective_risk(all_probs, fake_quality, all_labels)
    coverages.append(cov); risks.append(risk)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(coverages, risks, marker="o", linewidth=2, color="#3b82f6", markersize=4)
ax.fill_between(coverages, risks, alpha=0.15, color="#3b82f6")
ax.set_xlabel("Coverage", color="white", fontsize=12)
ax.set_ylabel("Selective Risk", color="white", fontsize=12)
ax.set_title("Coverage vs Selective Risk — Reject Option", color="white", fontsize=13)
ax.set_xlim(0,1); ax.set_ylim(0, max(risks)+0.05)
ax.tick_params(colors="white"); ax.set_facecolor("#1e1e2e")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(SAVE_DIR/"05_coverage_risk.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

rejector.c_threshold = 0.7
cov_70  = rejector.compute_coverage(all_probs, fake_quality)
risk_70 = rejector.compute_selective_risk(all_probs, fake_quality, all_labels)
print(f"\n📊 Selective @ c=0.70: Coverage={cov_70:.3f} | Risk={risk_70:.3f}")

# %% [markdown]
# ## 10. Error Analysis — Misclassified Samples

# %%
wrong_mask  = all_preds != all_labels
wrong_probs = all_probs[wrong_mask]
wrong_true  = all_labels[wrong_mask]
wrong_preds = all_preds[wrong_mask]
wrong_confs = wrong_probs.max(axis=1)

high_conf_errors = wrong_confs > 0.8
print(f"Total errors          : {wrong_mask.sum()}")
print(f"High-conf errors >80% : {high_conf_errors.sum()} ({high_conf_errors.mean()*100:.1f}% of errors)")

# Misclassification heatmap
n_cls         = len(EMOTION_CLASSES)
error_matrix  = np.zeros((n_cls, n_cls), dtype=int)
for t, p in zip(wrong_true, wrong_preds):
    error_matrix[t][p] += 1

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(error_matrix, annot=True, fmt="d", cmap="Reds",
            xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES, ax=ax)
ax.set_title("Misclassification Heatmap (True → Predicted)", color="white", fontsize=12)
ax.set_xlabel("Predicted as", color="white"); ax.set_ylabel("True Label", color="white")
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig(SAVE_DIR/"06_misclassification_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# Confidence distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(wrong_confs, bins=40, color="#ef4444", alpha=0.8, label="Wrong predictions")
ax.hist(all_probs[~wrong_mask].max(axis=1), bins=40, color="#22c55e", alpha=0.5, label="Correct predictions")
ax.axvline(wrong_confs.mean(), color="#f97316", linestyle="--", linewidth=2,
           label=f"Mean(wrong)={wrong_confs.mean():.2f}")
ax.set_title("Confidence: Correct vs Wrong (Calibrated Model)", color="white")
ax.set_xlabel("Max Confidence", color="white"); ax.set_ylabel("Count", color="white")
ax.legend(facecolor="#1e1e2e", labelcolor="white"); ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(SAVE_DIR/"07_confidence_distribution.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# %% [markdown]
# ## 11. FPS & Latency Benchmark

# %%
avg_lat = np.mean(latencies); fps_est = 1000.0 / avg_lat if avg_lat > 0 else 0
p50_lat = np.percentile(latencies, 50)
p95_lat = np.percentile(latencies, 95)

print("=" * 60)
print("⚡ LATENCY & FPS BENCHMARK (CPU)")
print("=" * 60)
print(f"  Avg Latency   : {avg_lat:.2f} ms/image")
print(f"  P50 Latency   : {p50_lat:.2f} ms")
print(f"  P95 Latency   : {p95_lat:.2f} ms")
print(f"  Estimated FPS : {fps_est:.1f} fps")

# %% [markdown]
# ## 12. Export Laporan Lengkap & Ringkasan Akhir

# %%
# Load WOA params jika ada
woa_params_path = ROOT / "checkpoints" / "woa_best_params.json"
woa_info = {}
if woa_params_path.exists():
    with open(woa_params_path) as f:
        woa_data = json.load(f)
    woa_info = {
        "best_params":  woa_data.get("best_params", {}),
        "best_fitness": woa_data.get("best_fitness", 0),
    }

# Per-class metrics
cm_final      = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
cm_norm_final = cm_final.astype(float) / (cm_final.sum(axis=1, keepdims=True) + 1e-6)
per_class_acc = cm_norm_final.diagonal()

summary = {
    "model": {
        "checkpoint":   str(ckpt_path),
        "backbone":     cfg.model.backbone,
        "num_classes":  cfg.model.num_classes,
        "temperature":  float(calibrator.temperature.item()) if calibrator else 1.0,
    },
    "woa_optimization": woa_info,
    "dataset": {
        "name":         cfg.dataset.name,
        "test_samples": int(len(test_ds)),
    },
    "classification": {
        "accuracy":  round(float(metrics["accuracy"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall":    round(float(metrics["recall"]), 4),
        "f1":        round(float(metrics["f1"]), 4),
        "UAR":       round(float(metrics["UAR"]), 4),
    },
    "per_class_accuracy": {
        EMOTION_CLASSES[i]: round(float(per_class_acc[i]), 4)
        for i in range(len(EMOTION_CLASSES))
    },
    "calibration": {k: round(float(v), 4) for k, v in cal_metrics.items()},
    "selective_prediction": {
        "coverage_at_c07":      round(cov_70, 4),
        "risk_at_c07":          round(risk_70, 4),
        "total_errors":         int(wrong_mask.sum()),
        "high_conf_errors_pct": round(float(high_conf_errors.mean()*100), 2),
    },
    "performance": {
        "device":         "cpu",
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "estimated_fps":  round(fps_est, 1),
    },
}

report_path = REPORT_DIR / "evaluation_summary.json"
with open(report_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Evaluation report saved: {report_path}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("🏆 FINAL EVALUATION SUMMARY")
print("=" * 65)
print(f"\n📊 Classification (WOA-Optimized + Calibrated, Test Set):")
print(f"   Accuracy    : {metrics['accuracy']*100:.2f}%")
print(f"   UAR (macro) : {metrics['UAR']*100:.2f}%")
print(f"   F1-Score    : {metrics['f1']*100:.2f}%")

print(f"\n📐 Per-Class Accuracy:")
for i, emo in enumerate(EMOTION_CLASSES):
    bar  = "█" * int(per_class_acc[i] * 20)
    icon = "✓" if per_class_acc[i] >= 0.65 else "⚠"
    print(f"   {icon} {emo:<12}: {per_class_acc[i]*100:>5.1f}%  [{bar:<20}]")

print(f"\n🌡️  Calibration:")
print(f"   ECE         : {cal_metrics['ECE']:.4f}")
print(f"   Brier Score : {cal_metrics['BrierScore']:.4f}")

print(f"\n❌ Reject Option @ c=0.7:")
print(f"   Coverage    : {cov_70*100:.1f}%")
print(f"   Risk        : {risk_70*100:.1f}%")

print(f"\n⚡ Performance (CPU):")
print(f"   Latency     : {avg_lat:.2f} ms/image")
print(f"   FPS         : {fps_est:.0f}")

if woa_info:
    print(f"\n🐋 WOA Optimization:")
    print(f"   Best fitness (quick F1): {woa_info.get('best_fitness', 0):.4f}")
    for k, v in woa_info.get("best_params", {}).items():
        print(f"   {k:<20}: {v:.6f}")

print(f"\n📁 Semua plot   : results/plots/evaluation/")
print(f"📋 Laporan JSON : results/reports/evaluation_summary.json")
print("\n🎉 Project evaluation complete!")
