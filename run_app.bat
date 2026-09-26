@echo off
title RAG-CRAW CLI
echo ========================================================
echo Running RAG-CRAW CLI app.py...
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
