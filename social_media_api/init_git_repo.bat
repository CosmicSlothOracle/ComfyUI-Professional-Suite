@echo off
echo Initializing Git repository for Social Media Video Generation API...
echo.

REM Check if Git is installed
where git >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Git is not installed or not in PATH. Please install Git.
    pause
    exit /b 1
)

REM Initialize Git repository
git init

REM Add .gitignore
git add .gitignore

REM Add README and documentation files
git add README.md
git add CONTRIBUTING.md
git add LICENSE
git add docs\architecture.md

REM Add configuration files
git add config\dashboard_config.json
git add config\google_credentials.json.example
git add pytest.ini
git add requirements.txt
git add setup.py

REM Add source code
git add api_server\
git add app\
git add comfy_integration\
git add __init__.py
git add run_api.py
git add run_api.bat

REM Add examples
git add examples\

REM Add tests
git add tests\

REM Add GitHub workflows
git add .github\

REM Make initial commit
git commit -m "Initial commit: Social Media Video Generation API"

echo.
echo Git repository initialized successfully!
echo.
echo Next steps:
echo 1. Create a repository on GitHub
echo 2. Run: git remote add origin https://github.com/yourusername/social-media-video-generator.git
echo 3. Run: git push -u origin main
echo.

pause