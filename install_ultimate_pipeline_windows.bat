@echo off
echo ========================================
echo ULTIMATE SPRITE PIPELINE - WINDOWS INSTALLER
echo ========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

:: Check if virtual environment exists, create if not
if not exist "venv310" (
    echo Creating virtual environment...
    python -m venv venv310
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv310\Scripts\activate.bat

echo.
echo ========================================
echo INSTALLING WINDOWS-COMPATIBLE PACKAGES
echo ========================================
echo.

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support (Windows compatible)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install requirements with retry logic
echo.
echo Installing requirements (attempt 1/3)...
pip install -r requirements_ultimate.txt
if errorlevel 1 (
    echo.
    echo First attempt failed, trying again with --no-deps flag...
    pip install -r requirements_ultimate.txt --no-deps
    if errorlevel 1 (
        echo.
        echo Second attempt failed, installing core packages individually...

        :: Install core packages one by one
        pip install numpy opencv-python Pillow scipy
        pip install transformers diffusers accelerate
        pip install rembg
        pip install ultralytics
        pip install tqdm rich colorama
        pip install requests aiohttp websocket-client websockets
        pip install pandas matplotlib seaborn
        pip install imageio tifffile
        pip install jupyter ipywidgets
        pip install psutil

        echo Core packages installed. Some optional packages may have been skipped.
    )
)

echo.
echo ========================================
echo RUNNING PIPELINE INSTALLER
echo ========================================
echo.

:: Run the Python installer
python install_ultimate_pipeline.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo INSTALLATION COMPLETED WITH WARNINGS
    echo ========================================
    echo Some packages may have failed to install, but the pipeline should still work
    echo with reduced functionality. You can try installing missing packages manually.
    echo.
) else (
    echo.
    echo ========================================
    echo INSTALLATION COMPLETED SUCCESSFULLY!
    echo ========================================
    echo.
)

echo To run the Ultimate Sprite Pipeline:
echo   1. Activate virtual environment: venv310\Scripts\activate
echo   2. Run pipeline: python ultimate_sprite_pipeline.py
echo.

pause