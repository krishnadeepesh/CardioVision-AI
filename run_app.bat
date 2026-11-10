@echo off
title CardioVision AI Launcher
echo ========================================================
echo Starting CardioVision AI Web Application
echo Workspace: D:\Data Science
echo Storage & Cache: D: Drive (C: Drive space protected)
echo ========================================================

set PIP_CACHE_DIR=D:\Data Science\.pip_cache
set TEMP=D:\Data Science\.tmp
set TMP=D:\Data Science\.tmp
set KERAS_HOME=D:\Data Science\.keras
set TORCH_HOME=D:\Data Science\.torch
set KERAS_BACKEND=torch

cd /d "D:\Data Science"

echo Activating environment and starting Streamlit...
"D:\Data Science\venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false

pause
