@echo off
set DISTUTILS_USE_SDK=1
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
set PYTHONPATH=E:\FreeToken\python
cd /d E:\FreeToken
> _setup_sched.py echo from setuptools import setup
>> _setup_sched.py echo from pathlib import Path
>> _setup_sched.py echo from torch.utils.cpp_extension import BuildExtension, CppExtension
>> _setup_sched.py echo.
>> _setup_sched.py echo ROOT = Path("E:/FreeToken")
>> _setup_sched.py echo.
>> _setup_sched.py echo setup(
>> _setup_sched.py echo     ext_modules=[
>> _setup_sched.py echo         CppExtension(
>> _setup_sched.py echo             name="freetoken.scheduler._freetoken_sched",
>> _setup_sched.py echo             sources=[
>> _setup_sched.py echo                 "python/freetoken/scheduler/csrc/sched_index.cpp",
>> _setup_sched.py echo                 "python/freetoken/scheduler/csrc/pybind_module.cpp",
>> _setup_sched.py echo             ],
>> _setup_sched.py echo             include_dirs=[str(ROOT / "python" / "freetoken" / "scheduler" / "csrc")],
>> _setup_sched.py echo             extra_compile_args=["/O2", "/std:c++17", "/EHsc"],
>> _setup_sched.py echo         ),
>> _setup_sched.py echo     ],
>> _setup_sched.py echo     cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
>> _setup_sched.py echo )

C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\python.exe _setup_sched.py build_ext --inplace --build-lib E:\FreeToken\python\freetoken\scheduler --build-temp E:\FreeToken\build\temp_sched > _build_sched.log 2>&1
echo EXIT=%ERRORLEVEL% >> _build_sched.log
del _setup_sched.py
echo === sched build complete ===