@echo off
echo Starting Social Media Video Generation API...
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist venv\installed.txt (
    echo Installing requirements...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
    echo > venv\installed.txt
)

REM Run the API
echo Starting API server...
python run_api.py

REM If we get here, the API has stopped
echo.
echo API server has stopped.
pause