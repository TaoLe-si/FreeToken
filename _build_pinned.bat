@echo off
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set DISTUTILS_USE_SDK=1
set PYTHON=C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\python.exe
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 1>nul
cd /d E:\FreeToken
%PYTHON% setup.py build_ext --inplace --build-lib E:\FreeToken\python 2>&1