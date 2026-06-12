@echo off
echo ============================================
echo  Konversi .py ke .ipynb
echo ============================================
echo.

set PYTHON=d:\BIG-Data\tuber\.venv\Scripts\python.exe
set NOTEBOOKS_DIR=d:\BIG-Data\tuber\notebooks

echo Mengkonversi notebooks...
%PYTHON% %NOTEBOOKS_DIR%\convert_notebooks.py

echo.
echo ============================================
echo  Konversi selesai!
echo  Buka notebook dengan:
echo  jupyter notebook notebooks/
echo ============================================
pause
