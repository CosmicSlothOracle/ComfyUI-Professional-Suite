@echo off
title Ultimate Sprite Processing Pipeline
color 0A

echo.
echo    ===============================================================
echo    🚀 ULTIMATE SPRITE PROCESSING PIPELINE
echo    ===============================================================
echo    State-of-the-art AI-powered sprite processing
echo    Perfect background removal • AI analysis • Intelligent upscaling
echo    ===============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ and add it to PATH.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if the pipeline file exists
if not exist "ultimate_sprite_pipeline.py" (
    echo ❌ Pipeline not found! Make sure you're in the correct directory.
    echo Expected file: ultimate_sprite_pipeline.py
    pause
    exit /b 1
)

REM Check if input directory exists
if not exist "input" (
    echo 📁 Creating input directory...
    mkdir input
    echo ✅ Input directory created. Place your sprite images in the 'input' folder.
    echo.
)

REM Count files in input directory
set /a file_count=0
for %%f in (input\*.png input\*.jpg input\*.jpeg) do set /a file_count+=1

if %file_count%==0 (
    echo ⚠️  No sprite images found in the 'input' directory!
    echo.
    echo 📋 Please add sprite images (PNG, JPG, JPEG) to the 'input' folder.
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✅ Found %file_count% sprite image(s) ready for processing.
echo.

:menu
echo ===============================================================
echo 🎯 QUALITY PRESETS
echo ===============================================================
echo.
echo 1. ⚡ FAST       - 30s per sprite  (CPU-friendly, good quality)
echo 2. ⚖️  BALANCED   - 60s per sprite  (balanced speed and quality)
echo 3. 🎯 PROFESSIONAL - 120s per sprite (excellent quality) ⭐ RECOMMENDED
echo 4. 🌟 ULTIMATE    - 300s per sprite (maximum quality, requires GPU)
echo.
echo 5. 🔧 CUSTOM OPTIONS
echo 6. 📖 HELP & INFO
echo 7. 🚪 EXIT
echo.
echo ===============================================================

set /p choice="Select quality preset (1-7): "

if "%choice%"=="1" goto fast
if "%choice%"=="2" goto balanced
if "%choice%"=="3" goto professional
if "%choice%"=="4" goto ultimate
if "%choice%"=="5" goto custom
if "%choice%"=="6" goto help
if "%choice%"=="7" goto exit

echo ❌ Invalid choice. Please select 1-7.
echo.
goto menu

:fast
echo.
echo ⚡ Starting FAST processing...
echo 📊 Estimated time: %file_count% × 30s = %file_count%0 seconds
echo.
python ultimate_sprite_pipeline.py --quality fast
goto results

:balanced
echo.
echo ⚖️ Starting BALANCED processing...
echo 📊 Estimated time: %file_count% × 60s = %file_count%00 seconds
echo.
python ultimate_sprite_pipeline.py --quality balanced
goto results

:professional
echo.
echo 🎯 Starting PROFESSIONAL processing...
echo 📊 Estimated time: %file_count% × 120s = %file_count%20 seconds
echo.
python ultimate_sprite_pipeline.py --quality professional
goto results

:ultimate
echo.
echo 🌟 Starting ULTIMATE processing...
echo 📊 Estimated time: %file_count% × 300s = %file_count%00 seconds
echo ⚠️  This requires a GPU with 4GB+ VRAM for best performance.
echo.
set /p confirm="Continue with Ultimate quality? (y/N): "
if /i not "%confirm%"=="y" goto menu
python ultimate_sprite_pipeline.py --quality ultimate
goto results

:custom
echo.
echo ===============================================================
echo 🔧 CUSTOM OPTIONS
echo ===============================================================
echo.
echo 1. Process specific directory
echo 2. CPU-only mode (no GPU)
echo 3. Custom quality + specific input
echo 4. Installation check
echo 5. Back to main menu
echo.
set /p custom_choice="Select option (1-5): "

if "%custom_choice%"=="1" goto custom_dir
if "%custom_choice%"=="2" goto cpu_mode
if "%custom_choice%"=="3" goto custom_quality
if "%custom_choice%"=="4" goto install_check
if "%custom_choice%"=="5" goto menu

echo ❌ Invalid choice.
goto custom

:custom_dir
echo.
set /p input_dir="Enter input directory path (or press Enter for 'input'): "
if "%input_dir%"=="" set input_dir=input
echo.
echo 🎯 Select quality for directory "%input_dir%":
echo 1. Fast  2. Balanced  3. Professional  4. Ultimate
set /p quality_choice="Quality (1-4): "

if "%quality_choice%"=="1" set quality=fast
if "%quality_choice%"=="2" set quality=balanced
if "%quality_choice%"=="3" set quality=professional
if "%quality_choice%"=="4" set quality=ultimate

if "%quality%"=="" (
    echo ❌ Invalid quality choice.
    goto custom_dir
)

echo.
echo 🚀 Processing directory "%input_dir%" with %quality% quality...
python ultimate_sprite_pipeline.py --input "%input_dir%" --quality %quality%
goto results

:cpu_mode
echo.
echo 💻 CPU-only mode (no GPU acceleration)
echo Recommended for systems without NVIDIA GPU or low VRAM.
echo.
echo 1. Fast CPU    2. Balanced CPU    3. Professional CPU
set /p cpu_choice="Select (1-3): "

if "%cpu_choice%"=="1" set cpu_quality=fast
if "%cpu_choice%"=="2" set cpu_quality=balanced
if "%cpu_choice%"=="3" set cpu_quality=professional

if "%cpu_quality%"=="" (
    echo ❌ Invalid choice.
    goto cpu_mode
)

echo.
echo 💻 Starting CPU-only processing with %cpu_quality% quality...
python ultimate_sprite_pipeline.py --quality %cpu_quality% --device cpu
goto results

:custom_quality
echo.
set /p custom_input="Input directory: "
set /p custom_qual="Quality (fast/balanced/professional/ultimate): "
echo.
echo 🚀 Processing "%custom_input%" with %custom_qual% quality...
python ultimate_sprite_pipeline.py --input "%custom_input%" --quality %custom_qual%
goto results

:install_check
echo.
echo 🔧 Checking installation...
python install_ultimate_pipeline.py --check
echo.
pause
goto menu

:help
echo.
echo ===============================================================
echo 📖 HELP & INFORMATION
echo ===============================================================
echo.
echo 🎯 QUALITY PRESETS:
echo.
echo FAST (30s per sprite)
echo   • U2Net background removal
echo   • OpenCV upscaling
echo   • Basic processing
echo   • CPU-friendly
echo.
echo BALANCED (60s per sprite)
echo   • BRIA RMBG background removal
echo   • Real-ESRGAN upscaling
echo   • Some AI analysis
echo   • Good speed/quality balance
echo.
echo PROFESSIONAL (120s per sprite) ⭐ RECOMMENDED
echo   • BiRefNet-HR background removal
echo   • SUPIR upscaling
echo   • Full AI analysis
echo   • Excellent quality
echo.
echo ULTIMATE (300s per sprite)
echo   • Multiple model ensemble
echo   • AI inpainting
echo   • Advanced motion analysis
echo   • Maximum quality
echo.
echo 📊 OUTPUT FORMATS:
echo   1. Individual frames (1x) - Perfect transparency
echo   2. Individual frames (2x) - Upscaled with transparency
echo   3. Animated GIF (1x) - Intelligent timing
echo   4. Animated GIF (2x) - Upscaled with intelligent timing
echo.
echo 📁 OUTPUT LOCATION:
echo   • output/ultimate_sprites/your_sprite_name/
echo.
echo 💡 TIPS:
echo   • Use Professional quality for best results
echo   • Ensure sprites have good contrast with background
echo   • Minimum recommended size: 256x256 pixels
echo   • GPU with 4GB+ VRAM recommended for Ultimate quality
echo.
echo ===============================================================
pause
goto menu

:results
echo.
echo ===============================================================
echo 🎉 PROCESSING COMPLETE!
echo ===============================================================
echo.
echo 📁 Results saved to: output\ultimate_sprites\
echo.
echo 📊 OUTPUT FILES:
echo   ✅ Individual frames (1x) - frames_1x folder
echo   ✅ Individual frames (2x) - frames_2x folder
echo   ✅ Animated GIF (1x) - sprite_name_animated.gif
echo   ✅ Animated GIF (2x) - sprite_name_animated_2x.gif
echo.
echo 📋 Processing report: processing_report.json
echo.

REM Ask to open output folder
set /p open_folder="🔍 Open output folder? (y/N): "
if /i "%open_folder%"=="y" (
    if exist "output\ultimate_sprites" (
        explorer "output\ultimate_sprites"
    ) else (
        echo ❌ Output folder not found.
    )
)

echo.
echo ===============================================================
set /p action="🔄 Process more sprites (P) or Exit (E)? [P/E]: "
if /i "%action%"=="p" goto menu
if /i "%action%"=="" goto menu
goto exit

:exit
echo.
echo ===============================================================
echo 👋 Thank you for using Ultimate Sprite Processing Pipeline!
echo.
echo 🔗 For support and updates:
echo    • GitHub: github.com/your-repo/ultimate-sprite-pipeline
echo    • Documentation: README_ULTIMATE_PIPELINE.md
echo.
echo ⭐ If this helped you, please consider starring the repository!
echo ===============================================================
echo.
pause
exit /b 0