@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup\windows.ps1 first. See docs\windows.md.
    popd
    pause
    exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" demo\app.py %*
set HAIRDECK_EXIT_CODE=%ERRORLEVEL%
popd
if not "%HAIRDECK_EXIT_CODE%"=="0" pause
endlocal & exit /b %HAIRDECK_EXIT_CODE%
