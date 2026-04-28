@echo off
REM Image Caption Generator GUI - Windows Launcher
REM This script makes it easy to launch the application on Windows

setlocal enabledelayedexpansion

cls
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                                                            ║
echo ║     Image Caption Generator - Windows Launcher            ║
echo ║                                                            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.7+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo.

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo ✓ Virtual environment found
    echo.
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
    echo.
) else (
    echo ⚠ Virtual environment not found
    echo.
    set /p create_venv="Create virtual environment now? (y/n): "
    if /i "!create_venv!"=="y" (
        echo Creating virtual environment...
        python -m venv venv
        echo ✓ Virtual environment created
        call venv\Scripts\activate.bat
        echo ✓ Virtual environment activated
    )
    echo.
)

REM Check for required files
echo Checking required files...
set missing_files=0

if not exist "model_19.h5" (
    echo ✗ model_19.h5 - MISSING
    set /a missing_files+=1
) else (
    echo ✓ model_19.h5 - Found
)

if not exist "prepro_by_raj.txt" (
    echo ✗ prepro_by_raj.txt - MISSING
    set /a missing_files+=1
) else (
    echo ✓ prepro_by_raj.txt - Found
)

echo.

if !missing_files! gtr 0 (
    echo ⚠ Warning: !missing_files! required file(s) missing
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" (
        exit /b 1
    )
    echo.
)

REM Check for requirements.txt
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo ✓ Dependencies installed
) else (
    echo ⚠ requirements.txt not found
    echo Installing core packages manually...
    python -m pip install --upgrade pip -q
    python -m pip install tensorflow keras gradio pillow numpy pyttsx3 -q
    echo ✓ Core packages installed
)

echo.
echo ════════════════════════════════════════════════════════════
echo Select Application Version:
echo ════════════════════════════════════════════════════════════
echo.
echo 1) Full Version (with text-to-speech, recommended)
echo 2) Simple Version (lightweight, fast)
echo 3) Run Tests
echo 4) Exit
echo.

set /p choice="Enter your choice (1-4): "

if "!choice!"=="1" (
    echo.
    echo Starting Full Version with text-to-speech...
    echo.
    python app_gui.py
) else if "!choice!"=="2" (
    echo.
    echo Starting Simple Version...
    echo.
    python app_gui_simple.py
) else if "!choice!"=="3" (
    echo.
    echo Running system tests...
    echo.
    python test_components.py
    echo.
    set /p rerun="Run application? (y/n): "
    if /i "!rerun!"=="y" (
        goto select_version
    )
) else if "!choice!"=="4" (
    echo.
    echo Exiting...
    goto end
) else (
    echo Invalid choice. Please try again.
    timeout /t 2 >nul
    goto select_version
)

:select_version
set /p again="Run another application? (y/n): "
if /i "!again!"=="y" (
    cls
    goto select_version
)

:end
echo.
echo Thank you for using Image Caption Generator!
echo.
pause
