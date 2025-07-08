@echo off
REM ================================================================
REM ComfyUI Sprite-Sheet Workflow - Vollautomatische Installation
REM ================================================================

echo.
echo ========================================================
echo  ComfyUI Sprite-Sheet Workflow Installation
echo ========================================================
echo.

REM Prüfe ob Python installiert ist
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python ist nicht installiert oder nicht im PATH!
    echo Bitte installiere Python 3.10+ von https://python.org
    pause
    exit /b 1
)

REM Prüfe ob Git installiert ist
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git ist nicht installiert oder nicht im PATH!
    echo Bitte installiere Git von https://git-scm.com
    pause
    exit /b 1
)

echo ✅ Python und Git gefunden

REM Erstelle Verzeichnisstruktur
echo.
echo 📁 Erstelle Verzeichnisstruktur...
mkdir custom_nodes 2>nul
mkdir models\checkpoints 2>nul
mkdir models\controlnet 2>nul
mkdir models\loras 2>nul
mkdir models\vae 2>nul
mkdir models\clip_vision 2>nul
mkdir models\ipadapter 2>nul
mkdir models\upscale_models 2>nul
mkdir input\sprite_sheets 2>nul
mkdir output\processed_sprites 2>nul
mkdir workflows\sprite_processing 2>nul

REM Installiere Custom Nodes
echo.
echo 🔧 Installiere Custom Nodes...
cd custom_nodes

echo    📦 ComfyUI-Manager...
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

echo    📦 ComfyUI-AnimateDiff-Evolved...
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git

echo    📦 ComfyUI-Advanced-ControlNet...
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git

echo    📦 ComfyUI-IPAdapter-Plus...
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

echo    📦 ComfyUI-Impact-Pack...
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git

echo    📦 ComfyUI-KJNodes...
git clone https://github.com/kijai/ComfyUI-KJNodes.git

echo    📦 ComfyUI-Essentials...
git clone https://github.com/cubiq/ComfyUI_essentials.git

echo    📦 ComfyUI-PixelArt-Detector...
git clone https://github.com/dimtoneff/ComfyUI-PixelArt-Detector.git

echo    📦 ComfyUI-MTB...
git clone https://github.com/melMass/comfy_mtb.git

cd ..

REM Erstelle Workflow-Dateien
echo.
echo 📋 Erstelle Workflow-Dateien...

REM Sprite Extractor Workflow
echo {"workflow": {"nodes": {"1": {"type": "LoadImage", "inputs": {"image": "sprite_sheet.png"}}, "2": {"type": "UnpackFrames", "inputs": {"images": ["1", 0], "frame_width": 64, "frame_height": 64, "frame_count": 0}}, "3": {"type": "DWPreprocessor", "inputs": {"image": ["2", 0], "detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}}, "4": {"type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "extracted_poses/"}}}}} > workflows\sprite_processing\sprite_extractor.json

REM Style Transfer Workflow
echo {"workflow": {"nodes": {"1": {"type": "LoadImage", "inputs": {"image": "character_frame.png"}}, "2": {"type": "LoadImage", "inputs": {"image": "pose_reference.png"}}, "3": {"type": "ControlNetLoader", "inputs": {"control_net_name": "control_v11p_sd15_openpose_fp16.safetensors"}}, "4": {"type": "ControlNetApplyAdvanced", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "control_net": ["3", 0], "image": ["2", 0], "strength": 0.8, "start_percent": 0.0, "end_percent": 1.0}}, "5": {"type": "CLIPTextEncode", "inputs": {"text": "high quality character art, detailed clothing, fantasy armor, vibrant colors", "clip": ["7", 1]}}, "6": {"type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, distorted, deformed", "clip": ["7", 1]}}, "7": {"type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "dreamshaperXL_v21TurboDPMSDE.safetensors"}}, "8": {"type": "KSampler", "inputs": {"model": ["7", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["9", 0], "seed": 42, "steps": 30, "cfg": 7.0, "sampler_name": "euler_a", "scheduler": "normal", "denoise": 1.0}}, "9": {"type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}, "10": {"type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["7", 2]}}, "11": {"type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "styled_character/"}}}}} > workflows\sprite_processing\style_transfer.json

REM Erstelle Startup-Script
echo.
echo 🚀 Erstelle Startup-Script...

echo @echo off > start_sprite_workflow.bat
echo echo Starting ComfyUI Sprite Processing Environment... >> start_sprite_workflow.bat
echo echo ================================================ >> start_sprite_workflow.bat
echo. >> start_sprite_workflow.bat
echo cd /d "%%~dp0" >> start_sprite_workflow.bat
echo. >> start_sprite_workflow.bat
echo REM Aktiviere Virtual Environment falls vorhanden >> start_sprite_workflow.bat
echo if exist "venv310\Scripts\activate.bat" ^( >> start_sprite_workflow.bat
echo     call venv310\Scripts\activate.bat >> start_sprite_workflow.bat
echo     echo ✅ Virtual Environment aktiviert >> start_sprite_workflow.bat
echo ^) >> start_sprite_workflow.bat
echo. >> start_sprite_workflow.bat
echo REM Starte ComfyUI mit optimierten Einstellungen >> start_sprite_workflow.bat
echo echo 🚀 Starte ComfyUI... >> start_sprite_workflow.bat
echo python main.py --listen --port 8188 --preview-method auto >> start_sprite_workflow.bat
echo. >> start_sprite_workflow.bat
echo pause >> start_sprite_workflow.bat

REM Erstelle README-Dateien
echo.
echo 📄 Erstelle Dokumentation...

echo Sprite-Sheets hier ablegen > input\sprite_sheets\README.txt
echo Unterstützte Formate: PNG, JPG, GIF >> input\sprite_sheets\README.txt
echo Empfohlene Auflösung: 512x512 oder höher >> input\sprite_sheets\README.txt

echo Verarbeitete Sprite-Sheets > output\processed_sprites\README.txt
echo Automatisch generierte Ausgaben des Workflows >> output\processed_sprites\README.txt

REM Erstelle Konfigurations-Datei
echo {
echo   "sprite_settings": {
echo     "default_frame_size": [64, 64],
echo     "supported_formats": ["png", "jpg", "gif"],
echo     "output_format": "png"
echo   },
echo   "style_presets": {
echo     "anime": {
echo       "lora": "anime_style_xl.safetensors",
echo       "strength": 0.8,
echo       "prompt_prefix": "anime style, vibrant colors, "
echo     },
echo     "pixel_art": {
echo       "lora": "pixel_art_xl_v1.5.safetensors",
echo       "strength": 1.0,
echo       "prompt_prefix": "pixel art, 8bit style, retro, "
echo     },
echo     "realistic": {
echo       "lora": "ACE++_1.0.safetensors",
echo       "strength": 0.6,
echo       "prompt_prefix": "photorealistic, detailed, "
echo     }
echo   },
echo   "processing_settings": {
echo     "batch_size": 4,
echo     "steps": 30,
echo     "cfg_scale": 7.0,
echo     "sampler": "euler_a"
echo   }
echo } > config.json

echo.
echo ================================================================
echo 🎉 INSTALLATION ERFOLGREICH ABGESCHLOSSEN!
echo ================================================================
echo.
echo 📋 Nächste Schritte:
echo.
echo 1. Modelle herunterladen:
echo    - Besuche https://civitai.com und lade folgende Modelle herunter:
echo      * DreamShaper XL v2.1 Turbo → models\checkpoints\
echo      * RealVisXL v4.0 → models\checkpoints\
echo      * Pixel Art XL v1.5 LoRA → models\loras\
echo      * Anime Style XL LoRA → models\loras\
echo.
echo 2. ControlNet-Modelle herunterladen:
echo    - Von https://huggingface.co/lllyasviel/ControlNet-v1-1:
echo      * control_v11p_sd15_openpose_fp16.safetensors → models\controlnet\
echo      * control_v11f1p_sd15_depth_fp16.safetensors → models\controlnet\
echo.
echo 3. VAE-Modell herunterladen:
echo    - Von https://huggingface.co/stabilityai/sd-vae-ft-mse-original:
echo      * vae-ft-mse-840000-ema-pruned.ckpt → models\vae\
echo.
echo 4. ComfyUI starten:
echo    - Doppelklick auf start_sprite_workflow.bat
echo.
echo 5. Beim ersten Start:
echo    - Gehe zu ComfyUI Manager und installiere fehlende Dependencies
echo    - Lade die erstellten Workflows aus workflows\sprite_processing\
echo.
echo 💡 Tipps:
echo - Lege deine Sprite-Sheets in input\sprite_sheets\
echo - Verwende die vorgefertigten Workflows für beste Ergebnisse
echo - Konfiguration anpassen in config.json
echo.
echo Viel Erfolg mit deinem Sprite-Processing-Workflow! 🎨
echo.
pause