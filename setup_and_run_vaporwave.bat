@echo off
echo.
echo ===============================================
echo     VAPORWAVE VIDEO PROCESSOR - SETUP
echo ===============================================
echo.

:: Wechsel in ComfyUI_engine Verzeichnis
cd /d "C:\Users\Public\ComfyUI-master\ComfyUI_engine"

echo 🔧 Überprüfe Eingabedatei...
if not exist "input\comica1750462002773.mp4" (
    echo ❌ Eingabedatei nicht gefunden!
    echo    Erwarteter Pfad: input\comica1750462002773.mp4
    pause
    exit /b 1
)
echo ✅ Eingabedatei gefunden

echo.
echo 📁 Erstelle Output-Verzeichnisse...
mkdir "output\vaporwave_gifs" 2>nul
mkdir "output\vaporwave_frames" 2>nul
mkdir "output\vaporwave_preview" 2>nul
echo ✅ Verzeichnisse erstellt

echo.
echo 🚀 Starte ComfyUI Server...
echo    (Das kann einen Moment dauern...)
echo.

:: Starte ComfyUI im Hintergrund
start /b python main.py --listen --port 8188

:: Warte auf Server-Start
echo ⏳ Warte auf Server-Initialisierung...
timeout /t 10 /nobreak >nul

echo.
echo 🎬 Führe Vaporwave-Workflow aus...
python run_vaporwave_workflow.py

echo.
echo ===============================================
echo     VAPORWAVE PROCESSING ABGESCHLOSSEN
echo ===============================================
echo.
echo 📁 Überprüfe die Ergebnisse in:
echo    • output\vaporwave_gifs\
echo    • output\vaporwave_frames\
echo.
echo 🌐 ComfyUI Web-Interface: http://localhost:8188
echo.

pause