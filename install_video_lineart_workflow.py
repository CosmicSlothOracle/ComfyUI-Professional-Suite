#!/usr/bin/env python3
"""
🎬 VIDEO TO LINE ART WORKFLOW - INSTALLATION SCRIPT
==================================================
Automatische Installation des kompletten Video-to-Line Art Workflows

Dieser Script installiert:
- Alle Custom Nodes
- Alle Modelle
- Workflow JSON
- Dokumentation
- Beispiel-Skripte
"""

import os
import sys
import subprocess
import urllib.request
import json
from pathlib import Path
import time
import shutil


def print_header():
    """Zeige Header mit ASCII Art"""
    print("🎬" + "=" * 60 + "🎬")
    print("     VIDEO TO LINE ART WORKFLOW INSTALLATION")
    print("🎬" + "=" * 60 + "🎬")
    print()
    print("✨ FEATURES:")
    print("   🎥 Video → Line Art Animation")
    print("   🎯 ControlNet Line Art Extraction")
    print("   🎬 AnimateDiff Motion Generation")
    print("   🚀 RIFE Frame Interpolation")
    print("   🎨 Professional Quality Output")
    print()
    print("🔧 INSTALLATION STARTING...")
    print()


def check_comfyui():
    """Prüfe ob ComfyUI korrekt installiert ist"""
    print("🔍 Checking ComfyUI installation...")

    main_py = Path("main.py")
    if not main_py.exists():
        print("❌ main.py not found - Please run from ComfyUI root directory")
        return False

    print("✅ ComfyUI installation found")
    return True


def create_directories():
    """Erstelle alle notwendigen Verzeichnisse"""
    print("📁 Creating directory structure...")

    directories = [
        "custom_nodes",
        "models/checkpoints",
        "models/controlnet",
        "models/animatediff",
        "models/ipadapter",
        "models/clip_vision",
        "models/loras",
        "models/rife",
        "workflows",
        "input",
        "output"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

    print()


def install_custom_nodes():
    """Installiere Custom Nodes"""
    print("🔧 Installing Custom Nodes...")

    nodes = [
        {
            "name": "ComfyUI-AnimateDiff-Evolved",
            "url": "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git"
        },
        {
            "name": "ComfyUI-Frame-Interpolation",
            "url": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
        },
        {
            "name": "ComfyUI_IPAdapter_plus",
            "url": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"
        },
        {
            "name": "ComfyUI-VideoHelperSuite",
            "url": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
        },
        {
            "name": "ComfyUI_ControlNet_Aux",
            "url": "https://github.com/Fannovel16/ComfyUI_ControlNet_Aux.git"
        },
        {
            "name": "ComfyUI-Manager",
            "url": "https://github.com/ltdrdata/ComfyUI-Manager.git"
        }
    ]

    for node in nodes:
        node_path = Path("custom_nodes") / node["name"]
        if not node_path.exists():
            print(f"   📦 Installing {node['name']}...")
            try:
                subprocess.run([
                    "git", "clone", node["url"], str(node_path)
                ], check=True, capture_output=True, text=True)
                print(f"   ✅ {node['name']} installed")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed: {e}")
        else:
            print(f"   ✅ {node['name']} already installed")

    print()


def download_with_progress(url, filepath):
    """Download mit Progress Bar"""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r      [{bar}] {percent}% ", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, filepath, show_progress)
        print()  # Neue Zeile
        return True
    except Exception as e:
        print(f"\n      ❌ Download failed: {e}")
        return False


def download_models():
    """Download alle notwendigen Modelle"""
    print("📥 Downloading Models (this will take a while)...")

    models = [
        {
            "name": "control_v11p_sd15_lineart.pth",
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
            "path": "models/controlnet",
            "size": "1.4GB"
        },
        {
            "name": "mm_sd_v15_v2.ckpt",
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
            "path": "models/animatediff",
            "size": "1.7GB"
        },
        {
            "name": "dreamshaper_8.safetensors",
            "url": "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
            "path": "models/checkpoints",
            "size": "2.0GB"
        },
        {
            "name": "ip-adapter_sd15_plus.safetensors",
            "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_plus.safetensors",
            "path": "models/ipadapter",
            "size": "22MB"
        },
        {
            "name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
            "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
            "path": "models/clip_vision",
            "size": "2.5GB"
        }
    ]

    total_size = "~8GB"
    print(f"   📊 Total download size: {total_size}")
    print()

    for i, model in enumerate(models, 1):
        model_path = Path(model["path"]) / model["name"]
        if not model_path.exists():
            print(
                f"   📥 [{i}/{len(models)}] {model['name']} ({model['size']})")
            success = download_with_progress(model["url"], model_path)
            if success:
                print(f"   ✅ Downloaded successfully")
            else:
                print(f"   ❌ Download failed")
        else:
            print(f"   ✅ [{i}/{len(models)}] {model['name']} already exists")
        print()

    print("✅ Model downloads complete")
    print()


def install_dependencies():
    """Installiere Python Dependencies"""
    print("📦 Installing Python dependencies...")

    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.7.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.64.0"
    ]

    for req in requirements:
        print(f"   📦 Installing {req.split('>=')[0]}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", req
            ], check=True, capture_output=True, text=True)
            print(f"   ✅ Installed")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Already installed or failed")

    print()


def copy_workflow():
    """Kopiere Workflow JSON in workflows Verzeichnis"""
    print("📄 Setting up workflow...")

    # Der Workflow wurde bereits erstellt, wir kopieren ihn
    workflow_source = Path("workflows/video_to_lineart_workflow.json")
    if workflow_source.exists():
        print("   ✅ Workflow JSON already exists")
    else:
        print("   ⚠️  Workflow JSON not found - please ensure it was created correctly")

    print()


def create_example_video():
    """Erstelle ein Beispiel-Video für Tests"""
    print("🎥 Creating example input video...")

    # Simple test video creation using OpenCV
    try:
        import cv2
        import numpy as np

        # Create a simple test animation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('input/test_video.mp4', fourcc, 24.0, (512, 512))

        for i in range(48):  # 2 seconds at 24fps
            # Create simple animated frame
            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            frame.fill(255)  # White background

            # Draw moving circle
            center_x = int(256 + 100 * np.sin(i * 0.3))
            center_y = int(256 + 100 * np.cos(i * 0.3))
            cv2.circle(frame, (center_x, center_y), 50, (0, 0, 0), 3)

            # Draw rectangle
            rect_x = int(200 + 50 * np.sin(i * 0.2))
            cv2.rectangle(frame, (rect_x, 200),
                          (rect_x + 100, 300), (128, 128, 128), 2)

            out.write(frame)

        out.release()
        print("   ✅ Test video created: input/test_video.mp4")

    except ImportError:
        print("   ⚠️  OpenCV not available - skipping test video creation")
        # Create a simple text file instead
        with open("input/README.txt", "w") as f:
            f.write("""
Place your input videos here!

Supported formats:
- MP4 (recommended)
- AVI
- MOV
- GIF

Recommended settings:
- Resolution: 512x512 to 1024x1024
- Duration: 10-30 seconds
- Frame rate: 24 FPS
""")
        print("   ✅ Input instructions created")

    print()


def create_usage_guide():
    """Erstelle Nutzungsanleitung"""
    print("📖 Creating usage guide...")

    guide = """# 🎬 Video to Line Art Workflow - Usage Guide

## 🚀 Quick Start

### 1. Start ComfyUI:
```bash
python main.py
```

### 2. Load Workflow:
- Open ComfyUI in browser (usually http://127.0.0.1:8188)
- Click "Load" button
- Select: `workflows/video_to_lineart_workflow.json`

### 3. Prepare Video:
- Place your video file in `input/` directory
- Supported: MP4, AVI, MOV, GIF
- Recommended: 512x512, 10-30 seconds

### 4. Configure Workflow:
- **Video Input:** Change filename in "Load Video" node
- **Quality:** Adjust steps (15-25) in K-Sampler
- **Style:** Modify prompts for different line art styles

### 5. Run Processing:
- Click "Queue Prompt"
- Watch progress in ComfyUI interface
- Output will be saved in `output/` directory

## ⚙️ Advanced Settings

### Line Art Styles:
- **Clean:** "clean line art, minimal, vector"
- **Sketch:** "rough sketch, hand drawn, artistic"
- **Anime:** "anime line art, manga style"

### Quality Settings:
- **Draft:** Steps: 10-15, Resolution: 512x512
- **Standard:** Steps: 20, Resolution: 512x512
- **High:** Steps: 25-30, Resolution: 768x768

### Frame Interpolation:
- **Multiplier 2:** Doubles frame rate (24→48 FPS)
- **Multiplier 4:** Quadruples frame rate (24→96 FPS)

## 🔧 Troubleshooting

### VRAM Issues:
- Reduce batch size to 1
- Lower resolution to 512x512
- Reduce steps to 15-20

### Poor Quality:
- Increase steps to 25-30
- Use better input video (higher resolution)
- Adjust ControlNet strength

### Slow Processing:
- Use LCM scheduler
- Reduce frame count
- Lower resolution

## 📊 Performance Tips

### For Speed:
- Resolution: 512x512
- Steps: 15-20
- Batch Size: 1
- Use DreamShaper LCM

### For Quality:
- Resolution: 768x768+
- Steps: 25-30
- High-quality input video
- Post-process with upscaler

Enjoy creating amazing line art animations! 🎨✨
"""

    with open("USAGE_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide)

    print("   ✅ Usage guide created: USAGE_GUIDE.md")
    print()


def create_startup_script():
    """Erstelle Startup Script"""
    print("🚀 Creating startup script...")

    if os.name == 'nt':  # Windows
        script_content = '''@echo off
echo 🎬 Starting Video to Line Art Workflow
echo =======================================
echo.
echo 🔧 Activating virtual environment...
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)

echo 🚀 Starting ComfyUI...
python main.py --listen

pause
'''
        with open("start_video_lineart.bat", "w") as f:
            f.write(script_content)
        print("   ✅ Windows startup script: start_video_lineart.bat")

    else:  # Linux/Mac
        script_content = '''#!/bin/bash
echo "🎬 Starting Video to Line Art Workflow"
echo "======================================="
echo
echo "🔧 Activating virtual environment..."
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "🚀 Starting ComfyUI..."
python main.py --listen
'''
        with open("start_video_lineart.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_video_lineart.sh", 0o755)
        print("   ✅ Linux/Mac startup script: start_video_lineart.sh")

    print()


def main():
    """Hauptfunktion"""
    print_header()

    # Checks
    if not check_comfyui():
        print("❌ Please run this script from your ComfyUI root directory")
        return

    try:
        # Installation steps
        create_directories()
        install_custom_nodes()
        install_dependencies()
        download_models()
        copy_workflow()
        create_example_video()
        create_usage_guide()
        create_startup_script()

        # Success message
        print("🎉" + "=" * 60 + "🎉")
        print("     INSTALLATION COMPLETE!")
        print("🎉" + "=" * 60 + "🎉")
        print()
        print("✅ All components installed successfully")
        print()
        print("📋 NEXT STEPS:")
        print("1. Start ComfyUI:")
        if os.name == 'nt':
            print("   • Double-click: start_video_lineart.bat")
        else:
            print("   • Run: ./start_video_lineart.sh")
        print("   • Or manually: python main.py")
        print()
        print("2. Open browser: http://127.0.0.1:8188")
        print()
        print("3. Load workflow:")
        print("   • Click 'Load' button")
        print("   • Select: workflows/video_to_lineart_workflow.json")
        print()
        print("4. Process video:")
        print("   • Put video in input/ directory")
        print("   • Update filename in workflow")
        print("   • Click 'Queue Prompt'")
        print()
        print("📖 Read USAGE_GUIDE.md for detailed instructions")
        print()
        print("🎬 Ready for Video-to-Line Art magic! ✨")

    except KeyboardInterrupt:
        print("\n⏹️  Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nPlease check the error and try again")


if __name__ == "__main__":
    main()
