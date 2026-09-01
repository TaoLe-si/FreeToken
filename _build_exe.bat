@echo off
REM FreeToken 自动化编译脚本 - 双击运行即可
REM Usage: double-click this file

set PYINSTALLER=C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe
set VENV_PY=C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\python.exe
set PYTHONPATH=E:\FreeToken\python

cd /d E:\FreeToken

echo === 编译调试版 FreeTokenDbg.exe ===
%PYINSTALLER% --clean --noconfirm FreeTokenDbg.spec
if errorlevel 1 (
    echo BUILD DEBUG FAILED, errorlevel=%errorlevel%
    pause
    exit /b 1
)
echo DEBUG BUILD OK

echo.
echo === 编译生产版 FreeToken.exe ===
%PYINSTALLER% --clean --noconfirm FreeToken.spec
if errorlevel 1 (
    echo BUILD RELEASE FAILED, errorlevel=%errorlevel%
    pause
    exit /b 1
)
echo RELEASE BUILD OK

echo.
echo === 编译完成 ===
dir dist\FreeToken*.exe
pause
