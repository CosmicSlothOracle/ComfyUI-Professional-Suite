@echo off
title ComfyUI Sprite-Processing Workflow
echo.
echo 🎮 COMFYUI SPRITE-PROCESSING WORKFLOW
echo =====================================
echo.

REM Prüfe ob Python verfügbar ist
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python nicht gefunden! Bitte installiere Python 3.8+
    pause
    exit /b 1
)

REM Prüfe ob wir im richtigen Verzeichnis sind
if not exist "ComfyUI_engine\main.py" (
    echo ❌ ComfyUI_engine\main.py nicht gefunden!
    echo 💡 Starte dieses Script aus dem ComfyUI-master Verzeichnis
    pause
    exit /b 1
)

echo ✅ Python gefunden
echo ✅ ComfyUI-Engine gefunden
echo.

REM Prüfe Sprite-Sheets
echo 🔍 Prüfe Sprite-Sheet Format...
python sprite_format_checker.py

echo.
echo 🚀 STARTE COMFYUI...
echo.
echo 💡 WORKFLOW-ANLEITUNG:
echo    1. Browser öffnet sich automatisch (http://localhost:8188)
echo    2. Klicke "Load" und wähle: workflows/sprite_processing/walk_anime_workflow.json
echo    3. Alle Workflows sind für deine Sprite-Sheets vorkonfiguriert
echo    4. Klicke "Queue Prompt" um AI-Processing zu starten
echo.
echo 🎯 Verfügbare Workflows:
echo    📋 walk_anime_workflow.json      - Anime-Style Lauf-Animation
echo    📋 attack_pixel_art_workflow.json - Pixel-Art Kampf-Animation
echo    📋 idle_enhanced_workflow.json   - Enhanced Idle-Animation
echo    📋 jump_anime_workflow.json      - Anime-Style Sprung-Animation
echo    📋 intro_pixel_art_workflow.json - Pixel-Art Intro-Sequenz
echo.

REM Wechsle ins ComfyUI-Engine Verzeichnis und starte Server
echo Starte ComfyUI Server im CPU-Modus (für bessere Kompatibilität)...
echo ⚠️  CPU-Modus ist langsamer als GPU, aber funktioniert auf allen Systemen
echo.
cd ComfyUI_engine
python main.py --listen --port 8188 --cpu

echo.
echo ComfyUI wurde beendet.
pause
