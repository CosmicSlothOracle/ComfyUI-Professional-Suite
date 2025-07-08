#!/usr/bin/env python3
"""
ComfyUI Sprite-Sheet Workflow - Vollautomatisches Setup
=======================================================
Dieses Script installiert alle notwendigen Komponenten für den
automatisierten Sprite-Sheet-Processing-Workflow.
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path
import json
import shutil


class SpriteWorkflowSetup:
    def __init__(self):
        self.comfyui_dir = Path("ComfyUI-master")
        self.models_dir = self.comfyui_dir / "models"
        self.custom_nodes_dir = self.comfyui_dir / "custom_nodes"

        # Definiere alle notwendigen Verzeichnisse
        self.directories = {
            "checkpoints": self.models_dir / "checkpoints",
            "controlnet": self.models_dir / "controlnet",
            "loras": self.models_dir / "loras",
            "vae": self.models_dir / "vae",
            "clip_vision": self.models_dir / "clip_vision",
            "upscale_models": self.models_dir / "upscale_models",
            "animatediff_models": self.models_dir / "animatediff_models",
            "ipadapter": self.models_dir / "ipadapter"
        }

    def create_directories(self):
        """Erstelle alle notwendigen Verzeichnisse"""
        print("📁 Erstelle Verzeichnisstruktur...")
        for name, path in self.directories.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {name}: {path}")

    def install_custom_nodes(self):
        """Installiere alle notwendigen Custom Nodes"""
        print("\n🔧 Installiere Custom Nodes...")

        nodes = {
            "ComfyUI-AnimateDiff-Evolved": "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
            "ComfyUI-Advanced-ControlNet": "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git",
            "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
            "ComfyUI-Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
            "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
            "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
            "ComfyUI_essentials": "https://github.com/cubiq/ComfyUI_essentials.git",
            "ComfyUI-PixelArt-Detector": "https://github.com/dimtoneff/ComfyUI-PixelArt-Detector.git",
            "ComfyUI_Dave_CustomNode": "https://github.com/dave-code-ruiz/ComfyUI_Dave_CustomNode.git",
            "comfy_mtb": "https://github.com/melMass/comfy_mtb.git"
        }

        for name, url in nodes.items():
            node_path = self.custom_nodes_dir / name
            if not node_path.exists():
                print(f"   📦 Installiere {name}...")
                subprocess.run(["git", "clone", url, str(node_path)],
                               capture_output=True, text=True, cwd=self.custom_nodes_dir)
            else:
                print(f"   ✅ {name} bereits installiert")

    def download_models(self):
        """Lade alle notwendigen Modelle herunter"""
        print("\n📥 Lade Modelle herunter...")

        models = {
            # Base Models
            "checkpoints": {
                "dreamshaperXL_v21TurboDPMSDE.safetensors": "https://civitai.com/api/download/models/251662",
                "realvisxlV40.safetensors": "https://civitai.com/api/download/models/344398"
            },

            # ControlNet Models
            "controlnet": {
                "control_v11p_sd15_openpose_fp16.safetensors": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
                "control_v11f1p_sd15_depth_fp16.safetensors": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors",
                "FLUX.1-dev-ControlNet-Union-Pro.safetensors": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"
            },

            # LoRA Models für Style Transfer
            "loras": {
                "pixel_art_xl_v1.5.safetensors": "https://civitai.com/api/download/models/135931",
                "anime_style_xl.safetensors": "https://civitai.com/api/download/models/182974",
                "ACE++_1.0.safetensors": "https://civitai.com/api/download/models/489729"
            },

            # VAE Models
            "vae": {
                "vae-ft-mse-840000-ema-pruned.ckpt": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
                "sdxl_vae.safetensors": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
            },

            # CLIP Vision Models
            "clip_vision": {
                "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/pytorch_model.bin"
            },

            # IPAdapter Models
            "ipadapter": {
                "ip-adapter_sd15.safetensors": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
                "ip-adapter-plus_sd15.safetensors": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
            },

            # Upscaler Models
            "upscale_models": {
                "4x_ESRGAN.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "4x-UltraSharp.pth": "https://mega.nz/file/qZRBmaIY#noaccttqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
            }
        }

        for category, files in models.items():
            print(f"\n   📂 {category.upper()}:")
            target_dir = self.directories[category]

            for filename, url in files.items():
                file_path = target_dir / filename
                if not file_path.exists():
                    print(f"      📥 {filename}...")
                    self.download_file(url, file_path)
                else:
                    print(f"      ✅ {filename} bereits vorhanden")

    def download_file(self, url, destination):
        """Lade eine Datei herunter mit Progress-Anzeige"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(
                                f"\r         Progress: {progress:.1f}%", end="")

            print(f"\r         ✅ Download abgeschlossen")

        except Exception as e:
            print(f"\r         ❌ Fehler beim Download: {e}")

    def create_workflow_files(self):
        """Erstelle vorgefertigte Workflow-Dateien"""
        print("\n📋 Erstelle Workflow-Dateien...")

        workflows_dir = self.comfyui_dir / "workflows" / "sprite_processing"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # Sprite Sheet Extractor Workflow
        sprite_extractor_workflow = {
            "workflow": {
                "nodes": {
                    "1": {
                        "type": "LoadImage",
                        "inputs": {"image": "sprite_sheet.png"}
                    },
                    "2": {
                        "type": "UnpackFrames",
                        "inputs": {
                            "images": ["1", 0],
                            "frame_width": 64,
                            "frame_height": 64,
                            "frame_count": 0
                        }
                    },
                    "3": {
                        "type": "DWPreprocessor",
                        "inputs": {
                            "image": ["2", 0],
                            "detect_hand": "enable",
                            "detect_body": "enable",
                            "detect_face": "enable"
                        }
                    },
                    "4": {
                        "type": "SaveImage",
                        "inputs": {
                            "images": ["3", 0],
                            "filename_prefix": "extracted_poses/"
                        }
                    }
                }
            }
        }

        # Style Transfer Workflow
        style_transfer_workflow = {
            "workflow": {
                "nodes": {
                    "1": {
                        "type": "LoadImage",
                        "inputs": {"image": "character_frame.png"}
                    },
                    "2": {
                        "type": "LoadImage",
                        "inputs": {"image": "pose_reference.png"}
                    },
                    "3": {
                        "type": "ControlNetLoader",
                        "inputs": {"control_net_name": "control_v11p_sd15_openpose_fp16.safetensors"}
                    },
                    "4": {
                        "type": "ControlNetApplyAdvanced",
                        "inputs": {
                            "positive": ["5", 0],
                            "negative": ["6", 0],
                            "control_net": ["3", 0],
                            "image": ["2", 0],
                            "strength": 0.8,
                            "start_percent": 0.0,
                            "end_percent": 1.0
                        }
                    },
                    "5": {
                        "type": "CLIPTextEncode",
                        "inputs": {
                            "text": "high quality character art, detailed clothing, fantasy armor, vibrant colors",
                            "clip": ["7", 1]
                        }
                    },
                    "6": {
                        "type": "CLIPTextEncode",
                        "inputs": {
                            "text": "blurry, low quality, distorted, deformed",
                            "clip": ["7", 1]
                        }
                    },
                    "7": {
                        "type": "CheckpointLoaderSimple",
                        "inputs": {"ckpt_name": "dreamshaperXL_v21TurboDPMSDE.safetensors"}
                    },
                    "8": {
                        "type": "KSampler",
                        "inputs": {
                            "model": ["7", 0],
                            "positive": ["4", 0],
                            "negative": ["4", 1],
                            "latent_image": ["9", 0],
                            "seed": 42,
                            "steps": 30,
                            "cfg": 7.0,
                            "sampler_name": "euler_a",
                            "scheduler": "normal",
                            "denoise": 1.0
                        }
                    },
                    "9": {
                        "type": "EmptyLatentImage",
                        "inputs": {
                            "width": 512,
                            "height": 512,
                            "batch_size": 1
                        }
                    },
                    "10": {
                        "type": "VAEDecode",
                        "inputs": {
                            "samples": ["8", 0],
                            "vae": ["7", 2]
                        }
                    },
                    "11": {
                        "type": "SaveImage",
                        "inputs": {
                            "images": ["10", 0],
                            "filename_prefix": "styled_character/"
                        }
                    }
                }
            }
        }

        # Workflows speichern
        with open(workflows_dir / "sprite_extractor.json", 'w') as f:
            json.dump(sprite_extractor_workflow, f, indent=2)

        with open(workflows_dir / "style_transfer.json", 'w') as f:
            json.dump(style_transfer_workflow, f, indent=2)

        print("   ✅ Workflow-Dateien erstellt")

    def create_automation_scripts(self):
        """Erstelle Automatisierungs-Scripts"""
        print("\n🤖 Erstelle Automatisierungs-Scripts...")

        scripts_dir = self.comfyui_dir / "scripts" / "sprite_automation"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Batch Processing Script
        batch_script = '''#!/usr/bin/env python3
"""
Automatisierte Sprite-Sheet Verarbeitung
========================================
"""

import os
import sys
import json
import requests
from pathlib import Path

class SpriteProcessor:
    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.input_dir = Path("input/sprite_sheets")
        self.output_dir = Path("output/processed_sprites")

    def process_sprite_sheet(self, sprite_path, style="anime"):
        """Verarbeite ein Sprite-Sheet vollautomatisch"""

        # 1. Frames extrahieren
        extract_workflow = self.load_workflow("sprite_extractor.json")
        extract_workflow["workflow"]["nodes"]["1"]["inputs"]["image"] = str(sprite_path)

        # 2. Style Transfer anwenden
        style_workflow = self.load_workflow("style_transfer.json")

        # 3. Workflows ausführen
        self.execute_workflow(extract_workflow)
        self.execute_workflow(style_workflow)

    def load_workflow(self, filename):
        """Lade Workflow aus Datei"""
        with open(f"workflows/sprite_processing/{filename}", 'r') as f:
            return json.load(f)

    def execute_workflow(self, workflow):
        """Führe Workflow in ComfyUI aus"""
        response = requests.post(f"{self.comfyui_url}/prompt", json=workflow)
        return response.json()

if __name__ == "__main__":
    processor = SpriteProcessor()

    # Verarbeite alle Sprite-Sheets im Input-Verzeichnis
    for sprite_file in processor.input_dir.glob("*.png"):
        print(f"Verarbeite: {sprite_file.name}")
        processor.process_sprite_sheet(sprite_file)
'''

        with open(scripts_dir / "batch_processor.py", 'w') as f:
            f.write(batch_script)

        # Konfigurationsdatei erstellen
        config = {
            "sprite_settings": {
                "default_frame_size": [64, 64],
                "supported_formats": ["png", "jpg", "gif"],
                "output_format": "png"
            },
            "style_presets": {
                "anime": {
                    "lora": "anime_style_xl.safetensors",
                    "strength": 0.8,
                    "prompt_prefix": "anime style, vibrant colors, "
                },
                "pixel_art": {
                    "lora": "pixel_art_xl_v1.5.safetensors",
                    "strength": 1.0,
                    "prompt_prefix": "pixel art, 8bit style, retro, "
                },
                "realistic": {
                    "lora": "ACE++_1.0.safetensors",
                    "strength": 0.6,
                    "prompt_prefix": "photorealistic, detailed, "
                }
            },
            "processing_settings": {
                "batch_size": 4,
                "steps": 30,
                "cfg_scale": 7.0,
                "sampler": "euler_a"
            }
        }

        with open(scripts_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print("   ✅ Automatisierungs-Scripts erstellt")

    def create_startup_script(self):
        """Erstelle Startup-Script für ComfyUI"""
        print("\n🚀 Erstelle Startup-Script...")

        startup_script = '''@echo off
echo Starting ComfyUI Sprite Processing Environment...
echo ================================================

cd /d "%~dp0"

REM Aktiviere Virtual Environment falls vorhanden
if exist "venv310\\Scripts\\activate.bat" (
    call venv310\\Scripts\\activate.bat
    echo ✅ Virtual Environment aktiviert
)

REM Starte ComfyUI mit optimierten Einstellungen für Sprite Processing
echo 🚀 Starte ComfyUI...
python main.py --listen --port 8188 --cpu-vae --preview-method auto

pause
'''

        with open(self.comfyui_dir / "start_sprite_workflow.bat", 'w') as f:
            f.write(startup_script)

        print("   ✅ Startup-Script erstellt")

    def setup_input_output_structure(self):
        """Erstelle Input/Output Verzeichnisstruktur"""
        print("\n📁 Erstelle Input/Output Struktur...")

        structure = {
            "input/sprite_sheets": "Originalsprite-Sheets hier ablegen",
            "input/style_references": "Stil-Referenzbilder hier ablegen",
            "input/pose_references": "Pose-Referenzen hier ablegen",
            "output/extracted_frames": "Extrahierte Einzelframes",
            "output/processed_sprites": "Verarbeitete Sprite-Sheets",
            "output/style_tests": "Stil-Tests und Variationen",
            "temp/processing": "Temporäre Verarbeitungsdateien"
        }

        for path, description in structure.items():
            full_path = self.comfyui_dir / path
            full_path.mkdir(parents=True, exist_ok=True)

            # Erstelle README in jedem Verzeichnis
            readme_path = full_path / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"{description}\n")
                f.write("=" * len(description) + "\n\n")
                if "input" in path:
                    f.write("Unterstützte Formate: PNG, JPG, GIF\n")
                    f.write("Empfohlene Auflösung: 512x512 oder höher\n")

            print(f"   ✅ {path}")

    def create_example_files(self):
        """Erstelle Beispiel-Dateien für Tests"""
        print("\n📄 Erstelle Beispiel-Dateien...")

        examples_dir = self.comfyui_dir / "examples" / "sprite_processing"
        examples_dir.mkdir(parents=True, exist_ok=True)

        # Beispiel-Prompts
        example_prompts = {
            "anime_character.txt": [
                "1girl, anime style, colorful outfit, fantasy armor, detailed face",
                "masterpiece, high quality, vibrant colors, dynamic pose",
                "character concept art, clean lines, cel shading"
            ],
            "pixel_art_character.txt": [
                "pixel art, 8bit style, retro character, blocky design",
                "limited color palette, crisp edges, game sprite",
                "16bit era, classic video game character"
            ],
            "realistic_character.txt": [
                "photorealistic character, detailed textures, natural lighting",
                "high resolution, cinematic quality, professional artwork",
                "realistic proportions, detailed clothing and accessories"
            ]
        }

        for filename, prompts in example_prompts.items():
            with open(examples_dir / filename, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    f.write(prompt + "\n")

        # Beispiel-Konfiguration für verschiedene Sprite-Größen
        sprite_configs = {
            "32x32": {"frame_width": 32, "frame_height": 32, "grid_size": "8x8"},
            "64x64": {"frame_width": 64, "frame_height": 64, "grid_size": "4x4"},
            "128x128": {"frame_width": 128, "frame_height": 128, "grid_size": "2x2"}
        }

        with open(examples_dir / "sprite_sizes.json", 'w') as f:
            json.dump(sprite_configs, f, indent=2)

        print("   ✅ Beispiel-Dateien erstellt")

    def run_setup(self):
        """Führe komplettes Setup aus"""
        print("🎯 ComfyUI Sprite-Sheet Workflow Setup")
        print("=" * 50)

        try:
            self.create_directories()
            self.install_custom_nodes()
            self.download_models()
            self.create_workflow_files()
            self.create_automation_scripts()
            self.create_startup_script()
            self.setup_input_output_structure()
            self.create_example_files()

            print("\n" + "=" * 50)
            print("🎉 SETUP ERFOLGREICH ABGESCHLOSSEN!")
            print("=" * 50)
            print("\n📋 Nächste Schritte:")
            print("1. Starte ComfyUI mit: start_sprite_workflow.bat")
            print("2. Lade Sprite-Sheets in input/sprite_sheets/")
            print(
                "3. Führe Automatisierung aus: python scripts/sprite_automation/batch_processor.py")
            print("\n💡 Tipps:")
            print("- Beispiel-Workflows findest du in workflows/sprite_processing/")
            print("- Konfiguration anpassen in scripts/sprite_automation/config.json")
            print("- Dokumentation in examples/sprite_processing/")

        except Exception as e:
            print(f"\n❌ Fehler beim Setup: {e}")
            print("Bitte überprüfe die Fehlermeldung und versuche es erneut.")


if __name__ == "__main__":
    setup = SpriteWorkflowSetup()
    setup.run_setup()
