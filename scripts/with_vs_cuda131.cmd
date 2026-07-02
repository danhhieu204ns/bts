@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
set PATH=%CD%\.venv\Scripts;%CUDA_HOME%\bin;C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
set PYTHONPATH=src
%*
