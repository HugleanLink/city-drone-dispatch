@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m src.cli serve
) else (
    python -m src.cli serve
)
if errorlevel 1 pause
