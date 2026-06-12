# Cara Memindahkan Project ke Google Colab

Panduan ini dibuat untuk project `tuber` agar bisa dijalankan di Google Colab, terutama untuk training dengan GPU T4.

## 1. Yang Perlu Dipindahkan

Pindahkan folder/file berikut:

```text
configs/
dataset/
notebooks/
src/
requirements.txt
train.py
calibrate.py
evaluate.py
app.py
demo.py
```

Jangan ikut upload folder/file berikut karena tidak perlu dan bisa membuat upload lama:

```text
.venv/
__pycache__/
archive (1).zip
results/        # opsional, boleh tidak diupload
checkpoints/    # opsional, upload hanya kalau mau lanjut dari checkpoint lama
```

Struktur dataset yang dibutuhkan notebook training:

```text
dataset/
  train/
    angry/
    contempt/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    suprise/
  validation/
    angry/
    contempt/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    suprise/
  test/
    ...
```

## 2. Buat ZIP dari Windows

Buka PowerShell, lalu jalankan dari luar folder project:

```powershell
Compress-Archive -Path "D:\BIG-Data\tuber\configs", `
                      "D:\BIG-Data\tuber\dataset", `
                      "D:\BIG-Data\tuber\notebooks", `
                      "D:\BIG-Data\tuber\src", `
                      "D:\BIG-Data\tuber\requirements.txt", `
                      "D:\BIG-Data\tuber\train.py", `
                      "D:\BIG-Data\tuber\calibrate.py", `
                      "D:\BIG-Data\tuber\evaluate.py", `
                      "D:\BIG-Data\tuber\app.py", `
                      "D:\BIG-Data\tuber\demo.py" `
                 -DestinationPath "D:\BIG-Data\tuber_colab.zip" -Force
```

Jika ingin membawa checkpoint lama juga, tambahkan:

```powershell
"D:\BIG-Data\tuber\checkpoints"
```

## 3. Upload ke Google Drive

Upload file ini ke Google Drive:

```text
tuber_colab.zip
```

Contoh lokasi yang dipakai di panduan ini:

```text
MyDrive/tuber_colab.zip
```

## 4. Setup Runtime Google Colab

Di Google Colab:

1. Buka notebook baru atau upload `notebooks/04_training.ipynb`.
2. Pilih menu `Runtime`.
3. Pilih `Change runtime type`.
4. Set `Hardware accelerator` ke `T4 GPU`.
5. Klik `Save`.

Cek GPU:

```python
!nvidia-smi

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Jika masih `CUDA available: False`, berarti runtime belum memakai GPU.

## 5. Extract Project di Colab

Jalankan cell ini di awal Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Extract ZIP ke storage lokal Colab:

```python
!rm -rf /content/tuber
!mkdir -p /content/tuber
!unzip -q "/content/drive/MyDrive/tuber_colab.zip" -d /content/tuber
```

Cek isi folder:

```python
!ls -lah /content/tuber
!ls -lah /content/tuber/dataset
```

Kalau hasil extract membuat folder tambahan seperti `/content/tuber/tuber/...`, gunakan folder itu sebagai project root. Contoh:

```python
%cd /content/tuber/tuber/notebooks
```

Kalau isi folder langsung berisi `src`, `dataset`, `notebooks`, gunakan:

```python
%cd /content/tuber/notebooks
```

Ini penting karena `04_training.ipynb` memakai:

```python
ROOT = Path("..").resolve()
```

Jadi current directory harus berada di folder `notebooks/`.

## 6. Install Dependency

Colab biasanya sudah punya PyTorch GPU. Agar CUDA tidak rusak karena reinstall PyTorch, install dependency selain `torch` dan `torchvision`:

```python
%cd /content/tuber
!grep -v -E "^(torch|torchvision)" requirements.txt > requirements_colab.txt
!pip install -q -r requirements_colab.txt
```

Jika project root ada di `/content/tuber/tuber`, pakai:

```python
%cd /content/tuber/tuber
!grep -v -E "^(torch|torchvision)" requirements.txt > requirements_colab.txt
!pip install -q -r requirements_colab.txt
```

Setelah install, kembali ke folder notebook:

```python
%cd /content/tuber/notebooks
```

atau:

```python
%cd /content/tuber/tuber/notebooks
```

## 7. Jalankan Training

Buka dan jalankan:

```text
notebooks/04_training.ipynb
```

Notebook sudah diset ke mode cepat:

```python
FAST_TRAINING = True
```

Setting mode cepat saat ini:

```text
epochs: 5
time limit: 1.5 jam
image size: 160
train subset: 12000 sample
validation subset: 2000 sample
freeze backbone: True
```

Dengan T4 GPU, training harus jauh lebih cepat dibanding CPU lokal.

Jika ingin training lebih lengkap, ubah:

```python
FAST_TRAINING = False
```

Mode lengkap memakai:

```text
epochs: 20
time limit: 3 jam
full dataset
freeze backbone: False
```

## 8. Simpan Hasil ke Google Drive

Colab runtime bisa reset, jadi setelah training selesai simpan output ke Drive:

```python
!mkdir -p "/content/drive/MyDrive/tuber_outputs"
!cp -r /content/tuber/checkpoints "/content/drive/MyDrive/tuber_outputs/"
!cp -r /content/tuber/results "/content/drive/MyDrive/tuber_outputs/"
```

Jika project root ada di `/content/tuber/tuber`, pakai:

```python
!mkdir -p "/content/drive/MyDrive/tuber_outputs"
!cp -r /content/tuber/tuber/checkpoints "/content/drive/MyDrive/tuber_outputs/"
!cp -r /content/tuber/tuber/results "/content/drive/MyDrive/tuber_outputs/"
```

File penting setelah training:

```text
checkpoints/best_model.pth
checkpoints/logs/training_log.csv
results/plots/training/
```

## 9. Lanjut Calibration dan Evaluation

Setelah `04_training.ipynb` selesai, lanjutkan notebook:

```text
notebooks/05_calibration.ipynb
notebooks/06_evaluation.ipynb
```

Pastikan tetap berada di folder `notebooks/` sebelum menjalankan cell:

```python
%cd /content/tuber/notebooks
```

atau:

```python
%cd /content/tuber/tuber/notebooks
```

## 10. Troubleshooting

### `Device: CPU`

Artinya Colab belum memakai GPU.

Solusi:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
```

Lalu restart runtime dan cek lagi:

```python
!nvidia-smi
```

### `ModuleNotFoundError: No module named 'src'`

Current directory salah.

Solusi:

```python
%cd /content/tuber/notebooks
```

atau:

```python
%cd /content/tuber/tuber/notebooks
```

### `Directory not found: dataset/train`

Dataset belum berada di lokasi yang benar.

Notebook mencari dataset di:

```text
PROJECT_ROOT/dataset/train
PROJECT_ROOT/dataset/validation
```

Cek:

```python
!ls -lah /content/tuber/dataset
!ls -lah /content/tuber/dataset/train
!ls -lah /content/tuber/dataset/validation
```

### Training tetap lama

Gunakan setting cepat:

```python
FAST_TRAINING = True
```

Kalau masih lama, turunkan lagi:

```python
cfg["training"]["epochs"] = 3
cfg["training"]["max_train_samples"] = 6000
cfg["training"]["max_val_samples"] = 1000
cfg["dataset"]["image_size"] = 128
```

### Output hilang setelah runtime reset

Simpan checkpoint dan result ke Google Drive sebelum menutup Colab:

```python
!mkdir -p "/content/drive/MyDrive/tuber_outputs"
!cp -r /content/tuber/checkpoints "/content/drive/MyDrive/tuber_outputs/"
!cp -r /content/tuber/results "/content/drive/MyDrive/tuber_outputs/"
```
