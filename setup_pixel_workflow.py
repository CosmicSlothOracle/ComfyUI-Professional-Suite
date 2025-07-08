#!/usr/bin/env python3
"""
ComfyUI Pixel Art GIF to MP4 Setup Script
Automatisiert die Installation von Dependencies und Modellen
"""
import os
import sys
import requests
import subprocess
from pathlib import Path
import urllib.request
from tqdm import tqdm


class PixelWorkflowSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.models_dir = self.base_dir / "models"
        self.upscale_dir = self.models_dir / "upscale_models"
        self.loras_dir = self.models_dir / "loras"
        self.checkpoints_dir = self.models_dir / "checkpoints"

    def create_directories(self):
        """Erstellt erforderliche Verzeichnisse"""
        directories = [
            self.upscale_dir,
            self.loras_dir,
            self.checkpoints_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Verzeichnis erstellt: {directory}")

    def install_dependencies(self):
        """Installiert Python Dependencies"""
        dependencies = [
            'imageio>=2.31.1',
            'imageio-ffmpeg',
            'Pillow>=10.0.0',
            'tqdm>=4.64.0',
            'opencv-python>=4.8.0',
            'requests>=2.31.0'
        ]

        print("📦 Installiere Dependencies...")
        for dep in dependencies:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], capture_output=True, text=True, check=True)
                print(f"✓ Installiert: {dep}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Fehler bei {dep}: {e}")
                return False

        return True

    def download_with_progress(self, url, filepath):
        """Download mit Progress Bar"""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                bar_length = 50
                filled_length = int(bar_length * percent / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(
                    f'\r|{bar}| {percent:.1f}% {downloaded/1024/1024:.1f}MB', end='', flush=True)

        try:
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print()  # Neue Zeile nach Progress Bar
            return True
        except Exception as e:
            print(f"\n❌ Download-Fehler: {e}")
            return False

    def download_models(self):
        """Lädt erforderliche Modelle herunter"""
        models = {
            # RealESRGAN Upscaler
            'RealESRGAN_x4plus.pth': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'dir': self.upscale_dir,
                'size': '67MB'
            },
            'RealESRGAN_x2plus.pth': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                'dir': self.upscale_dir,
                'size': '67MB'
            }
        }

        print("🎨 Lade Upscaler-Modelle herunter...")

        for filename, info in models.items():
            filepath = info['dir'] / filename

            if filepath.exists():
                print(f"✓ Bereits vorhanden: {filename}")
                continue

            print(f"📥 Lade {filename} ({info['size']})...")
            if self.download_with_progress(info['url'], filepath):
                print(f"✓ Erfolgreich heruntergeladen: {filename}")
            else:
                return False

        return True

    def check_comfyui_nodes(self):
        """Prüft verfügbare ComfyUI Nodes"""
        video_nodes = self.base_dir / "comfy_extras" / "nodes_video.py"
        upscale_nodes = self.base_dir / "comfy_extras" / "nodes_upscale_model.py"

        checks = {
            "Video Nodes": video_nodes.exists(),
            "Upscale Nodes": upscale_nodes.exists()
        }

        print("🔍 ComfyUI Nodes Check:")
        for name, exists in checks.items():
            status = "✓" if exists else "❌"
            print(f"{status} {name}: {'Verfügbar' if exists else 'Nicht gefunden'}")

        return all(checks.values())

    def create_test_workflow(self):
        """Erstellt Test-Workflow JSON"""
        workflow = {
            "1": {
                "class_type": "LoadVideo",
                "inputs": {
                    "file": "test_input.gif"
                }
            },
            "2": {
                "class_type": "GetVideoComponents",
                "inputs": {
                    "video": ["1", 0]
                }
            },
            "3": {
                "class_type": "UpscaleModelLoader",
                "inputs": {
                    "model_name": "RealESRGAN_x4plus.pth"
                }
            },
            "4": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["3", 0],
                    "image": ["2", 0]
                }
            },
            "5": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["4", 0],
                    "fps": 24.0
                }
            },
            "6": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["5", 0],
                    "filename_prefix": "pixel_upscaled_test",
                    "format": "mp4"
                }
            }
        }

        workflow_file = self.base_dir / "pixel_workflow_test.json"
        import json
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)

        print(f"✓ Test-Workflow erstellt: {workflow_file}")
        return workflow_file

    def run_setup(self):
        """Führt komplettes Setup aus"""
        print("🚀 ComfyUI Pixel Workflow Setup startet...\n")

        # Schritt 1: Verzeichnisse
        self.create_directories()
        print()

        # Schritt 2: Dependencies
        if not self.install_dependencies():
            print("❌ Setup fehlgeschlagen bei Dependencies")
            return False
        print()

        # Schritt 3: Modelle downloaden
        if not self.download_models():
            print("❌ Setup fehlgeschlagen bei Modellen")
            return False
        print()

        # Schritt 4: ComfyUI Nodes prüfen
        if not self.check_comfyui_nodes():
            print(
                "⚠️  Einige ComfyUI Nodes fehlen - möglicherweise manuell nachinstallieren")
        print()

        # Schritt 5: Test-Workflow
        self.create_test_workflow()
        print()

        print("🎉 Setup erfolgreich abgeschlossen!")
        print("\nNächste Schritte:")
        print("1. ComfyUI starten")
        print("2. Test-Workflow laden: pixel_workflow_test.json")
        print("3. Ein GIF in den input/ Ordner legen")
        print("4. Workflow ausführen")

        return True


if __name__ == "__main__":
    setup = PixelWorkflowSetup()
    success = setup.run_setup()

    if not success:
        sys.exit(1)
