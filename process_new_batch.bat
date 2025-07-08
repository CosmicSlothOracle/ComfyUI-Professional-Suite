@echo off
echo ========================================
echo Processing Second Batch of Spritesheets
echo ========================================

cd /d "%~dp0"

echo Starting batch processing...
python process_new_batch.py

echo.
echo Processing complete! Check output/spritesheet_batch_2/ for results.
pause