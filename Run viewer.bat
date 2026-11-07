@echo off
cd /d "%~dp0viewer"
python -c "import numpy, zstandard, PySide6" 2>nul || python -m pip install -r requirements.txt
start "" pythonw viewer.py
