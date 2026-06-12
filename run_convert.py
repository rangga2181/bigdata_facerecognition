"""
run_convert.py
Script helper untuk konversi notebooks .py → .ipynb
Jalankan: python run_convert.py
Atau double-click dari explorer
"""
import subprocess
import sys
import os

def main():
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    script_path = os.path.join(os.path.dirname(__file__), "notebooks", "convert_notebooks.py")

    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    print(f"Using Python: {python_exe}")
    print(f"Converting notebooks...")

    result = subprocess.run([python_exe, script_path], capture_output=False)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
