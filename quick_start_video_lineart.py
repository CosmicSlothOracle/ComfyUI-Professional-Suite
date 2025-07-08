#!/usr/bin/env python3
"""
🎬 QUICK START - VIDEO TO LINE ART WORKFLOW
==========================================
Erstellt die essentiellen Dateien für den Video-to-Line Art Workflow
ohne große Model Downloads (für schnellen Start)
"""

import os
import sys
import json
from pathlib import Path


def print_header():
    """Zeige Header"""
    print("🎬" + "=" * 50 + "🎬")
    print("   QUICK START - VIDEO TO LINE ART WORKFLOW")
    print("🎬" + "=" * 50 + "🎬")
    print()
    print("🚀 Creating essential workflow files...")
    print("📝 Models müssen separat heruntergeladen werden")
    print()


def create_directories():
    """Erstelle Verzeichnisse"""
    print("📁 Creating directories...")

    directories = [
        "workflows",
        "input",
        "output",
        "models/checkpoints",
        "models/controlnet",
        "models/animatediff"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

    print()


def create_workflow_json():
    """Erstelle Workflow JSON"""
    print("📄 Creating workflow JSON...")

    workflow = {
        "last_node_id": 15,
        "last_link_id": 20,
        "nodes": [
            {
                "id": 1,
                "type": "VHS_LoadVideo",
                "pos": [100, 100],
                "size": [300, 100],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {
                        "name": "IMAGE",
                        "type": "IMAGE",
                        "links": [1]
                    }
                ],
                "properties": {
                    "Node name for S&R": "VHS_LoadVideo"
                },
                "widgets_values": [
                    "Schlossgifdance.mp4",
                    "image",
                    0, 0, 0, 24, 1, "Disabled"
                ],
                "title": "📹 Load Video"
            },
            {
                "id": 2,
                "type": "ControlNetLoader",
                "pos": [100, 250],
                "size": [300, 60],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {
                        "name": "CONTROL_NET",
                        "type": "CONTROL_NET",
                        "links": [2]
                    }
                ],
                "properties": {
                    "Node name for S&R": "ControlNetLoader"
                },
                "widgets_values": [
                    "control_v11p_sd15_lineart.pth"
                ],
                "title": "🎛️ ControlNet Line Art"
            },
            {
                "id": 3,
                "type": "CheckpointLoaderSimple",
                "pos": [100, 350],
                "size": [300, 100],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {
                        "name": "MODEL",
                        "type": "MODEL",
                        "links": [3]
                    },
                    {
                        "name": "CLIP",
                        "type": "CLIP",
                        "links": [4, 5]
                    },
                    {
                        "name": "VAE",
                        "type": "VAE",
                        "links": [6]
                    }
                ],
                "properties": {
                    "Node name for S&R": "CheckpointLoaderSimple"
                },
                "widgets_values": [
                    "dreamshaper_8.safetensors"
                ],
                "title": "🏗️ Base Model"
            },
            {
                "id": 4,
                "type": "CLIPTextEncode",
                "pos": [450, 200],
                "size": [400, 200],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {
                        "name": "clip",
                        "type": "CLIP",
                        "link": 4
                    }
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [7]
                    }
                ],
                "properties": {
                    "Node name for S&R": "CLIPTextEncode"
                },
                "widgets_values": [
                    "clean line art drawing, black and white, simple lines, minimal style, vector art, high contrast, no shading, cartoon animation, smooth lines, professional line art"
                ],
                "title": "📝 Positive Prompt"
            },
            {
                "id": 5,
                "type": "CLIPTextEncode",
                "pos": [450, 450],
                "size": [400, 150],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [
                    {
                        "name": "clip",
                        "type": "CLIP",
                        "link": 5
                    }
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [8]
                    }
                ],
                "properties": {
                    "Node name for S&R": "CLIPTextEncode"
                },
                "widgets_values": [
                    "blurry, low quality, noise, artifacts, photorealistic, color, shading, gradients, complex details, messy lines"
                ],
                "title": "🚫 Negative Prompt"
            },
            {
                "id": 6,
                "type": "LineArtPreprocessor",
                "pos": [900, 100],
                "size": [250, 150],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [
                    {
                        "name": "image",
                        "type": "IMAGE",
                        "link": 1
                    }
                ],
                "outputs": [
                    {
                        "name": "IMAGE",
                        "type": "IMAGE",
                        "links": [9]
                    }
                ],
                "properties": {
                    "Node name for S&R": "LineArtPreprocessor"
                },
                "widgets_values": [
                    512,
                    "disable"
                ],
                "title": "✏️ Line Art Preprocessor"
            },
            {
                "id": 7,
                "type": "ControlNetApply",
                "pos": [900, 300],
                "size": [300, 150],
                "flags": {},
                "order": 6,
                "mode": 0,
                "inputs": [
                    {
                        "name": "conditioning",
                        "type": "CONDITIONING",
                        "link": 7
                    },
                    {
                        "name": "control_net",
                        "type": "CONTROL_NET",
                        "link": 2
                    },
                    {
                        "name": "image",
                        "type": "IMAGE",
                        "link": 9
                    }
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [10]
                    }
                ],
                "properties": {
                    "Node name for S&R": "ControlNetApply"
                },
                "widgets_values": [
                    1.0
                ],
                "title": "🎯 ControlNet Apply"
            },
            {
                "id": 8,
                "type": "EmptyLatentImage",
                "pos": [900, 500],
                "size": [250, 100],
                "flags": {},
                "order": 7,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {
                        "name": "LATENT",
                        "type": "LATENT",
                        "links": [11]
                    }
                ],
                "properties": {
                    "Node name for S&R": "EmptyLatentImage"
                },
                "widgets_values": [
                    512, 512, 16
                ],
                "title": "🖼️ Empty Latent"
            },
            {
                "id": 9,
                "type": "KSampler",
                "pos": [1250, 150],
                "size": [300, 250],
                "flags": {},
                "order": 8,
                "mode": 0,
                "inputs": [
                    {
                        "name": "model",
                        "type": "MODEL",
                        "link": 3
                    },
                    {
                        "name": "positive",
                        "type": "CONDITIONING",
                        "link": 10
                    },
                    {
                        "name": "negative",
                        "type": "CONDITIONING",
                        "link": 8
                    },
                    {
                        "name": "latent_image",
                        "type": "LATENT",
                        "link": 11
                    }
                ],
                "outputs": [
                    {
                        "name": "LATENT",
                        "type": "LATENT",
                        "links": [12]
                    }
                ],
                "properties": {
                    "Node name for S&R": "KSampler"
                },
                "widgets_values": [
                    12345, "randomize", 20, 7.5, "euler_ancestral", "normal", 1.0
                ],
                "title": "🎲 K-Sampler"
            },
            {
                "id": 10,
                "type": "VAEDecode",
                "pos": [1250, 450],
                "size": [200, 100],
                "flags": {},
                "order": 9,
                "mode": 0,
                "inputs": [
                    {
                        "name": "samples",
                        "type": "LATENT",
                        "link": 12
                    },
                    {
                        "name": "vae",
                        "type": "VAE",
                        "link": 6
                    }
                ],
                "outputs": [
                    {
                        "name": "IMAGE",
                        "type": "IMAGE",
                        "links": [13, 14]
                    }
                ],
                "properties": {
                    "Node name for S&R": "VAEDecode"
                },
                "title": "🎨 VAE Decode"
            },
            {
                "id": 11,
                "type": "VHS_VideoCombine",
                "pos": [1600, 200],
                "size": [300, 200],
                "flags": {},
                "order": 10,
                "mode": 0,
                "inputs": [
                    {
                        "name": "images",
                        "type": "IMAGE",
                        "link": 13
                    }
                ],
                "outputs": [],
                "properties": {
                    "Node name for S&R": "VHS_VideoCombine"
                },
                "widgets_values": [
                    24, 0, "line_art_animation", "video/h264-mp4", True, True, False, False
                ],
                "title": "📹 Video Combine"
            },
            {
                "id": 12,
                "type": "PreviewImage",
                "pos": [1600, 450],
                "size": [300, 300],
                "flags": {},
                "order": 11,
                "mode": 0,
                "inputs": [
                    {
                        "name": "images",
                        "type": "IMAGE",
                        "link": 14
                    }
                ],
                "properties": {
                    "Node name for S&R": "PreviewImage"
                },
                "title": "👁️ Preview"
            }
        ],
        "links": [
            [1, 1, 0, 6, 0, "IMAGE"],
            [2, 2, 0, 7, 1, "CONTROL_NET"],
            [3, 3, 0, 9, 0, "MODEL"],
            [4, 3, 1, 4, 0, "CLIP"],
            [5, 3, 1, 5, 0, "CLIP"],
            [6, 3, 2, 10, 1, "VAE"],
            [7, 4, 0, 7, 0, "CONDITIONING"],
            [8, 5, 0, 9, 2, "CONDITIONING"],
            [9, 6, 0, 7, 2, "IMAGE"],
            [10, 7, 0, 9, 1, "CONDITIONING"],
            [11, 8, 0, 9, 3, "LATENT"],
            [12, 9, 0, 10, 0, "LATENT"],
            [13, 10, 0, 11, 0, "IMAGE"],
            [14, 10, 0, 12, 0, "IMAGE"]
        ],
        "groups": [
            {
                "title": "📹 Video Input",
                "bounding": [50, 50, 400, 200],
                "color": "#3f789e"
            },
            {
                "title": "🎛️ Control & Models",
                "bounding": [50, 270, 400, 350],
                "color": "#3f789e"
            },
            {
                "title": "✏️ Line Art Processing",
                "bounding": [870, 50, 350, 300],
                "color": "#a1309b"
            },
            {
                "title": "🎬 Generation",
                "bounding": [1220, 100, 350, 500],
                "color": "#88a96e"
            },
            {
                "title": "📹 Output",
                "bounding": [1570, 150, 350, 600],
                "color": "#b06634"
            }
        ],
        "config": {},
        "extra": {
            "ds": {
                "scale": 0.75,
                "offset": [0, 0]
            },
            "workspace_info": {
                "name": "Video to Line Art - Schlossgifdance",
                "description": "Convert Schlossgifdance.mp4 to line art animation",
                "version": "1.0",
                "author": "ComfyUI Video-to-Line Art Workflow"
            }
        },
        "version": 0.4
    }

    # Speichere Workflow JSON
    workflow_path = Path("workflows/video_to_lineart_schloss.json")
    workflow_path.parent.mkdir(exist_ok=True)

    with open(workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Workflow saved: {workflow_path}")
    print()


def create_model_download_script():
    """Erstelle Script zum Model Download"""
    print("📥 Creating model download script...")

    script = '''#!/usr/bin/env python3
"""
📥 MODEL DOWNLOAD SCRIPT
========================
Download benötigte Modelle für Video-to-Line Art Workflow
"""

import urllib.request
from pathlib import Path

def download_with_progress(url, filepath):
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar_length = 30
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\\r      [{bar}] {percent}% ", end="", flush=True)

    print(f"📥 Downloading {filepath.name}...")
    urllib.request.urlretrieve(url, filepath, show_progress)
    print()

def main():
    print("📥 DOWNLOADING MODELS FOR VIDEO-TO-LINE ART")
    print("=" * 50)

    models = [
        {
            "name": "control_v11p_sd15_lineart.pth",
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
            "path": "models/controlnet"
        },
        {
            "name": "dreamshaper_8.safetensors",
            "url": "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
            "path": "models/checkpoints"
        }
    ]

    for model in models:
        model_path = Path(model["path"]) / model["name"]
        if not model_path.exists():
            download_with_progress(model["url"], model_path)
            print(f"✅ {model['name']} downloaded")
        else:
            print(f"✅ {model['name']} already exists")

    print("\\n🎉 All models downloaded!")
    print("💡 You can now use the workflow in ComfyUI")

if __name__ == "__main__":
    main()
'''

    with open("download_models.py", "w", encoding="utf-8") as f:
        f.write(script)

    print("   ✅ Model download script: download_models.py")
    print()


def create_quick_guide():
    """Erstelle Schnellstart-Anleitung"""
    print("📖 Creating quick guide...")

    guide = '''# 🎬 Video-to-Line Art - Schnellstart

## 🚀 Sofort loslegen:

### 1. Modelle downloaden:
```bash
python download_models.py
```

### 2. ComfyUI starten:
```bash
python main.py
```

### 3. Workflow laden:
- Browser: http://127.0.0.1:8188
- Load → `workflows/video_to_lineart_schloss.json`

### 4. Video verarbeiten:
- Ihr Video ist bereits voreingestellt: `Schlossgifdance.mp4`
- Oder ändern Sie den Dateinamen im "Load Video" Node
- Queue Prompt klicken

## ⚙️ Parameter anpassen:

### Prompts:
- **Positive:** "clean line art, vector, minimal style"
- **Negative:** "blurry, color, shading, complex"

### Qualität:
- **Steps:** 15-25 (K-Sampler)
- **CFG:** 7.5 (Prompt-Stärke)
- **Resolution:** 512x512

## 📁 Dateien:
- **Input:** Ihr Video hier platzieren
- **Output:** Fertige Line Art Animation

## 🔧 Custom Nodes needed:
- ComfyUI-VideoHelperSuite
- ComfyUI_ControlNet_Aux

Viel Erfolg! 🎨✨
'''

    with open("QUICK_START.md", "w", encoding="utf-8") as f:
        f.write(guide)

    print("   ✅ Quick guide: QUICK_START.md")
    print()


def create_input_readme():
    """Erstelle README für Input Verzeichnis"""
    print("📁 Creating input directory README...")

    readme = '''# 📹 INPUT DIRECTORY

Place your videos here for processing!

## Supported Formats:
- ✅ MP4 (recommended)
- ✅ AVI
- ✅ MOV
- ✅ GIF

## Recommended Settings:
- **Resolution:** 512x512 to 1024x1024
- **Duration:** 5-30 seconds
- **Frame Rate:** 24-30 FPS
- **Quality:** High contrast, clear shapes

## Example Files:
- Your current video: `Schlossgifdance.mp4`
- Add more videos as needed

## Tips:
- Shorter videos = faster processing
- Good lighting improves line art quality
- Simple backgrounds work best
'''

    with open("input/README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("   ✅ Input README: input/README.md")
    print()


def main():
    """Hauptfunktion"""
    print_header()

    try:
        create_directories()
        create_workflow_json()
        create_model_download_script()
        create_quick_guide()
        create_input_readme()

        print("🎉" + "=" * 50 + "🎉")
        print("     QUICK START SETUP COMPLETE!")
        print("🎉" + "=" * 50 + "🎉")
        print()
        print("✅ Essential files created")
        print()
        print("📋 NEXT STEPS:")
        print("1. Download models: python download_models.py")
        print("2. Install Custom Nodes:")
        print("   - ComfyUI-VideoHelperSuite")
        print("   - ComfyUI_ControlNet_Aux")
        print("3. Start ComfyUI: python main.py")
        print("4. Load workflow: workflows/video_to_lineart_schloss.json")
        print("5. Process your video!")
        print()
        print("📖 See QUICK_START.md for details")
        print()
        print("🎬 Ready for Line Art creation! ✨")

    except Exception as e:
        print(f"❌ Setup failed: {e}")


if __name__ == "__main__":
    main()
