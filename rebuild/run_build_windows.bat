@echo off
setlocal EnableDelayedExpansion
call "D:\VS\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set TEMP=E:\tmp\neohorse
set TMP=E:\tmp\neohorse
if not exist "%TEMP%" mkdir "%TEMP%"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
set CUDA_PATH_V12_3=%CUDA_PATH%
set CUDAToolkit_ROOT=%CUDA_PATH%
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
REM Prepend toolkit bin + lib\x64 so linker/CMake prefer CUDA 12.3 over Anaconda cudart64_110
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\lib\x64;%PATH%
set LIB=%CUDA_PATH%\lib\x64;%LIB%
set LIBRARY_PATH=%CUDA_PATH%\lib\x64;%LIBRARY_PATH%
REM Drop Anaconda Lib\x64 from PATH/LIB if present (keep Scripts/bin later is fine; Lib\x64 must not win)
set "NEWPATH="
for %%P in ("%PATH:;=" "%") do (
  echo %%~P | findstr /i /c:"\anaconda\Lib\x64" /c:"\anaconda\Library\lib" >nul
  if errorlevel 1 (
    if defined NEWPATH (set "NEWPATH=!NEWPATH!;%%~P") else (set "NEWPATH=%%~P")
  )
)
if defined NEWPATH set "PATH=!NEWPATH!"
set "NEWLIB="
for %%P in ("%LIB:;=" "%") do (
  echo %%~P | findstr /i /c:"\anaconda\Lib\x64" /c:"\anaconda\Library\lib" >nul
  if errorlevel 1 (
    if defined NEWLIB (set "NEWLIB=!NEWLIB!;%%~P") else (set "NEWLIB=%%~P")
  )
)
if defined NEWLIB set "LIB=!NEWLIB!"
cd /d "E:\project\mousemove\neohorse-jev-gguf\runtime"
echo START %DATE% %TIME% > "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
echo CUDA_PATH=!CUDA_PATH!>> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
echo PATH_HEAD=!PATH:~0,200!>> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
echo LIB_HEAD=!LIB:~0,200!>> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
"E:\miniconda3\envs\ai\python.exe" -u build_windows.py --cuda-arch 89 --jobs 8 >> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log" 2>&1
echo EXIT=!ERRORLEVEL! >> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
echo END %DATE% %TIME% >> "E:\project\mousemove\neohorse-jev-gguf\build_windows.log"
exit /b !ERRORLEVEL!
