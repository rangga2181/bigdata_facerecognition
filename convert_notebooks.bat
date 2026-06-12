@echo off
echo =============================================
echo  FER Project - Convert Notebooks (.py -> .ipynb)
echo =============================================
echo.

REM Cari Python dari venv atau sistem
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo Using venv Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python
)

echo.
echo Converting notebooks...
%PYTHON% notebooks\convert_notebooks.py

echo.
echo =============================================
echo  Done!
echo =============================================
pause
