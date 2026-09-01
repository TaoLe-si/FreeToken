@echo off
REM Build _freetoken_igpu.pyd (FreeToken iGPU C++ glue module, 2026-08-30).
REM Prerequisites:
REM   - Microsoft Visual Studio 18 (Enterprise / Community / Build Tools)
REM   - CUDA Toolkit v13.0+ (cudart.lib in lib/x64/)
REM   - PyTorch 2.x in C:\Users\Administrator\AppData\Local\FreeToken\venv
REM
REM Usage:  _build_glue.bat

set DISTUTILS_USE_SDK=1
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
set PYTHONPATH=E:\FreeToken\python
cd /d E:\FreeToken

REM Write inline setup helper
> _setup_glue.py echo from setuptools import setup
>> _setup_glue.py echo from pathlib import Path
>> _setup_glue.py echo from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension
>> _setup_glue.py echo.
>> _setup_glue.py echo ROOT = Path('E:/FreeToken')
>> _setup_glue.py echo cuda_home = Path(CUDA_HOME)
>> _setup_glue.py echo cuda_lib = [str(cuda_home / 'lib64')]
>> _setup_glue.py echo if (cuda_home / 'lib').exists(): cuda_lib.append(str(cuda_home / 'lib'))
>> _setup_glue.py echo if (cuda_home / 'lib' / 'x64').exists(): cuda_lib.append(str(cuda_home / 'lib' / 'x64'))
>> _setup_glue.py echo.
>> _setup_glue.py echo setup(
>> _setup_glue.py echo     ext_modules=[
>> _setup_glue.py echo         CppExtension(
>> _setup_glue.py echo             name='freetoken.kernel._freetoken_igpu',
>> _setup_glue.py echo             sources=[
>> _setup_glue.py echo                 'python/freetoken/kernel/csrc/glue/igpu_service.cpp',
>> _setup_glue.py echo                 'python/freetoken/kernel/csrc/glue/mtp_head.cpp',
>> _setup_glue.py echo                 'python/freetoken/kernel/csrc/glue/pybind_module.cpp',
>> _setup_glue.py echo             ],
>> _setup_glue.py echo             include_dirs=[str(cuda_home / 'include')] + [
>> _setup_glue.py echo                 str(ROOT / 'python' / 'freetoken' / 'kernel' / 'csrc' / 'glue'),
>> _setup_glue.py echo             ],
>> _setup_glue.py echo             library_dirs=cuda_lib,
>> _setup_glue.py echo             libraries=['cudart'],
>> _setup_glue.py echo             extra_compile_args=['-O3', '-std=c++17', '-pthread'],
>> _setup_glue.py echo         ),
>> _setup_glue.py echo     ],
>> _setup_glue.py echo     cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
>> _setup_glue.py echo )

C:\Users\Administrator\AppData\Local\FreeToken\venv\Scripts\python.exe _setup_glue.py build_ext --inplace --build-lib E:\FreeToken\python\freetoken\kernel --build-temp E:\FreeToken\build\temp > _build_glue.log 2>&1
echo EXIT=%ERRORLEVEL% >> _build_glue.log
del _setup_glue.py
echo === build complete ===
