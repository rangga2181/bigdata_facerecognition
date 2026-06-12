"""
notebooks/convert_notebooks.py
Mengkonversi file .py (jupytext format) ke .ipynb menggunakan nbformat.

Usage:
    cd d:/BIG-Data/tuber
    python notebooks/convert_notebooks.py

Requirements:
    pip install nbformat
    (opsional: pip install jupytext)
"""
import re
import json
from pathlib import Path

try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    print("⚠️  nbformat not installed. Run: pip install nbformat")

NOTEBOOK_DIR = Path(__file__).parent
NOTEBOOKS = [
    "01_eda.py",
    "01b_undersampling.py",
    "02_preprocessing.py",
    "03_augmentation_imbalance.py",
    "04_training.py",
    "05_calibration.py",
    "06_evaluation.py",
]


def py_to_notebook(py_path: Path) -> dict:
    """
    Konversi file .py dengan cell markers ke format notebook dict.

    Cell markers yang didukung:
        # %%            → code cell
        # %% [markdown] → markdown cell
    """
    source = py_path.read_text(encoding="utf-8")
    lines  = source.split("\n")

    cells   = []
    current_type   = None
    current_lines  = []

    def flush_cell():
        if current_type is None or not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if not content:
            return

        if current_type == "markdown":
            # Hapus leading '# ' dari setiap baris
            md_lines = []
            for line in current_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            content = "\n".join(md_lines).strip()
            if content:
                cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            if content:
                cells.append(nbformat.v4.new_code_cell(content))

    for line in lines:
        stripped = line.strip()

        # Deteksi cell marker
        if stripped == "# %%":
            flush_cell()
            current_type  = "code"
            current_lines = []
        elif stripped == "# %% [markdown]":
            flush_cell()
            current_type  = "markdown"
            current_lines = []
        else:
            if current_type is not None:
                current_lines.append(line)
            # Baris sebelum marker pertama: diabaikan

    flush_cell()

    # Buat notebook
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language":     "python",
        "name":         "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.9.0",
    }
    return nb


def convert_all():
    if not NBFORMAT_AVAILABLE:
        print("❌ nbformat not available. Install with: pip install nbformat")
        return

    print("📓 Converting .py notebooks to .ipynb ...\n")
    success = 0

    for nb_file in NOTEBOOKS:
        py_path = NOTEBOOK_DIR / nb_file
        ipynb_path = py_path.with_suffix(".ipynb")

        if not py_path.exists():
            print(f"  ⏭️  Skipped (not found): {nb_file}")
            continue

        try:
            nb = py_to_notebook(py_path)
            with open(ipynb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            n_cells = len(nb.cells)
            print(f"  ✅ {nb_file} → {ipynb_path.name} ({n_cells} cells)")
            success += 1
        except Exception as e:
            print(f"  ❌ {nb_file}: {e}")

    print(f"\n✅ Done! Converted {success}/{len(NOTEBOOKS)} notebooks.")
    print(f"   Run: jupyter notebook notebooks/")


def try_jupytext_convert():
    """Alternative: gunakan jupytext jika tersedia (lebih akurat)."""
    import subprocess
    result = subprocess.run(["jupytext", "--version"], capture_output=True)
    if result.returncode != 0:
        return False

    print("\n🔄 Using jupytext for conversion (more accurate)...")
    for nb_file in NOTEBOOKS:
        py_path = NOTEBOOK_DIR / nb_file
        if py_path.exists():
            subprocess.run(["jupytext", "--to", "notebook", str(py_path)])
            print(f"  ✅ {nb_file} → {nb_file.replace('.py','.ipynb')}")
    return True


if __name__ == "__main__":
    # Coba jupytext dulu, fallback ke custom converter
    try:
        if not try_jupytext_convert():
            convert_all()
    except FileNotFoundError:
        convert_all()
