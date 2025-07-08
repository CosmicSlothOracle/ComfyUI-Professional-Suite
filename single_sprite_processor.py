#!/usr/bin/env python3
"""
SINGLE SPRITE PROCESSOR
Verarbeitet nur: C:\Users\Public\ComfyUI-master\input\sprite_sheets\idle_9_512x512_grid9x1.png
"""

import json
import os
import requests
from pathlib import Path


class SingleSpriteProcessor:
    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.sprite_path = Path(
            "input/sprite_sheets/idle_9_512x512_grid9x1.png")
        self.output_dir = Path("output/single_sprite_processing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def upload_sprite_to_comfyui(self):
        """Lädt das Idle-Sprite zu ComfyUI hoch"""
        if not self.sprite_path.exists():
            print(f"❌ Sprite nicht gefunden: {self.sprite_path}")
            return None

        try:
            with open(self.sprite_path, 'rb') as f:
                files = {'image': (self.sprite_path.name, f, 'image/png')}
                response = requests.post(
                    f"{self.comfyui_url}/upload/image", files=files)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name')
                print(f"✅ Sprite hochgeladen als: {uploaded_name}")
                return uploaded_name
            else:
                print(f"❌ Upload failed: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

    def create_img2img_workflow(self, uploaded_filename, style):
        """Erstellt IMG2IMG Workflow für das Idle-Sprite"""

        style_configs = {
            "anime": {
                "positive": "anime character sprite sheet, idle animation, cel shading, vibrant colors, 2d game art, clean lineart, consistent frame sizes, sprite grid",
                "negative": "blurry, low quality, 3d, realistic, inconsistent frames, merged characters, different sizes",
                "denoise": 0.6
            },
            "pixel": {
                "positive": "pixel art sprite sheet, idle animation, 16-bit style, retro game, crisp pixels, consistent frame grid, game sprites",
                "negative": "blurry, smooth, anti-aliased, high resolution, realistic, inconsistent grid",
                "denoise": 0.5
            },
            "enhanced": {
                "positive": "high quality sprite sheet, idle animation, detailed game art, consistent frames, professional sprites, clear grid layout",
                "negative": "blurry, low quality, inconsistent, merged frames, different frame sizes",
                "denoise": 0.7
            }
        }

        config = style_configs.get(style, style_configs["anime"])

        # IMG2IMG Workflow
        workflow = {
            "1": {
                "inputs": {
                    "image": uploaded_filename,
                    "upload": "image"
                },
                "class_type": "LoadImage"
            },
            "2": {
                "inputs": {
                    "ckpt_name": "sdxl.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "3": {
                "inputs": {
                    "text": config["positive"],
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": config["negative"],
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["2", 2]
                },
                "class_type": "VAEEncode"
            },
            "6": {
                "inputs": {
                    "seed": 42 + hash(style) % 1000000,
                    "steps": 12,  # Schneller
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": config["denoise"],  # Style-spezifisch
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "7": {
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["2", 2]
                },
                "class_type": "VAEDecode"
            },
            "8": {
                "inputs": {
                    "filename_prefix": f"IDLE_SPRITE_{style}_",
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def process_idle_sprite(self):
        """Verarbeitet das Idle-Sprite mit AI"""
        print("🎮 SINGLE SPRITE PROCESSOR")
        print(f"🎯 TARGET: {self.sprite_path}")
        print("=" * 60)

        # Server-Check
        try:
            response = requests.get(
                f"{self.comfyui_url}/system_stats", timeout=5)
            if response.status_code != 200:
                print("❌ ComfyUI Server nicht verfügbar!")
                return False
        except:
            print("❌ ComfyUI Server nicht erreichbar!")
            return False

        print("✅ ComfyUI Server ist bereit")

        # Sprite-Check
        if not self.sprite_path.exists():
            print(f"❌ SPRITE NICHT GEFUNDEN: {self.sprite_path}")
            print(f"💡 Vollständiger Pfad: {self.sprite_path.absolute()}")
            return False

        print(f"✅ Sprite gefunden: {self.sprite_path.name}")

        # Sprite-Info
        file_size = self.sprite_path.stat().st_size / 1024
        print(f"📏 Dateigröße: {file_size:.1f} KB")

        # Upload
        print("\n📤 UPLOAD ZU COMFYUI...")
        uploaded_filename = self.upload_sprite_to_comfyui()
        if not uploaded_filename:
            return False

        # AI-Verarbeitung für jeden Style
        styles = ["anime", "pixel", "enhanced"]
        successful = 0

        print(f"\n🤖 STARTE AI-VERARBEITUNG...")
        print("-" * 40)

        for style in styles:
            print(f"\n🎨 STYLE: {style.upper()}")

            try:
                # Workflow erstellen
                workflow = self.create_img2img_workflow(
                    uploaded_filename, style)

                # An ComfyUI senden
                response = requests.post(f"{self.comfyui_url}/prompt",
                                         json={"prompt": workflow},
                                         timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    print(f"✅ AI-Workflow gestartet (ID: {prompt_id})")
                    successful += 1
                else:
                    print(f"❌ Workflow-Fehler: {response.text[:200]}")

            except Exception as e:
                print(f"❌ Fehler bei {style}: {e}")

        print(f"\n" + "=" * 60)
        print("🏁 IDLE SPRITE VERARBEITUNG ABGESCHLOSSEN")
        print("=" * 60)
        print(f"✅ Gestartete AI-Workflows: {successful}/{len(styles)}")
        print(f"📁 Ergebnisse in: ComfyUI_engine/output/")
        print("🔍 Suche nach: IDLE_SPRITE_*.png")

        if successful > 0:
            print("\n🎉 AI VERARBEITET JETZT DEIN IDLE-SPRITE!")
            print("⏳ Warte ein paar Minuten...")
            print("🎯 Dein originales 9x1 Grid wird transformiert!")

        return successful > 0


def main():
    processor = SingleSpriteProcessor()
    processor.process_idle_sprite()


if __name__ == "__main__":
    main()
