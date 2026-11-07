@echo off
cd /d "%~dp0viewer"
python -c "import numpy, zstandard, PySide6" 2>/dev/null || python -m pip install -r requirements.txt
python viewer.py
pause
