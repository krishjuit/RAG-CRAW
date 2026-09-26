@echo off
title RAG-CRAW Web App
echo ========================================================
echo Starting RAG-CRAW Streamlit Web App...
echo ========================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\streamlit.exe" run client.py
) else (
    python -m streamlit run client.py
)
pause
