@echo off
REM ============================================
REM  FreeToken dev daemon (with iGPU/dense panel)
REM ============================================
set PYTHONPATH=E:\FreeToken\python
start "FreeToken daemon" cmd /k C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\python.exe -u -m freetoken.cli daemon --host 127.0.0.1 --port 1900 --log-level info
echo.
echo 控制面板:  http://127.0.0.1:1900/panel
echo 选项API:   http://127.0.0.1:1900/engine/options
pause