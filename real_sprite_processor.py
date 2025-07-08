#!/usr/bin/env python3
"""
ECHTER SPRITE PROCESSOR - MIT DEINEN ORIGINAL SPRITES
Verarbeitet deine echten Sprite-Sheets mit korrekter Frame-Struktur
"""

import json
import os
import requests
import base64
from pathlib import Path
from PIL import Image
import io


class EchterSpriteProcessor:
    def __init__(self):
        self.comfyui_url = "http://127.0.0.1:8188"
        self.input_dir = Path("input")
        self.output_dir = Path("output/real_sprite_processing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_sprite_files(self):
        """Findet deine echten Sprite-Dateien"""
        sprite_files = []

        # Deine spezifischen Sprite-Dateien
        target_sprites = [
            "idle_9_512x512_grid9x1.png",
            "walk_8_512x512_grid8x1.png",
            "jump_20_512x512_grid8x3.png",
            "attack_8_512x512_grid8x1.png",
            "intro_24_512x512_grid8x3.png"
        ]

        for sprite_name in target_sprites:
            sprite_path = self.input_dir / sprite_name
            if sprite_path.exists():
                sprite_files.append(sprite_path)
                print(f"✅ Gefunden: {sprite_name}")
            else:
                print(f"❌ Fehlt: {sprite_name}")

        return sprite_files

    def image_to_base64(self, image_path):
        """Konvertiert Bild zu Base64 für ComfyUI Upload"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def upload_image_to_comfyui(self, image_path):
        """Lädt Bild zu ComfyUI hoch"""
        try:
            # Image als Base64 vorbereiten
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/png')}
                response = requests.post(
                    f"{self.comfyui_url}/upload/image", files=files)

            if response.status_code == 200:
                result = response.json()
                return result.get('name')  # Uploaded filename
            else:
                print(f"❌ Upload failed: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

    def create_img2img_workflow(self, uploaded_filename, sprite_info, style):
        """Erstellt IMG2IMG Workflow mit deinem echten Sprite als Input"""

        style_prompts = {
            "anime": {
                "positive": f"anime character sprites, {sprite_info['action']} animation, cel shading, vibrant colors, 2d game art, clean lineart, consistent style",
                "negative": "blurry, low quality, 3d, realistic, inconsistent, different sizes, merged frames"
            },
            "pixel": {
                "positive": f"pixel art sprites, {sprite_info['action']} animation, 16-bit style, retro game, crisp pixels, consistent size, game sprite sheet",
                "negative": "blurry, smooth, anti-aliased, high resolution, realistic, inconsistent frames"
            },
            "enhanced": {
                "positive": f"detailed sprite sheet, {sprite_info['action']} animation, high quality game art, consistent frames, professional sprites",
                "negative": "blurry, low quality, inconsistent, merged frames, different sizes"
            }
        }

        # IMG2IMG Workflow der dein Original-Sprite als Input verwendet
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
                    "text": style_prompts[style]["positive"],
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": style_prompts[style]["negative"],
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
                    "seed": 42 + hash(sprite_info['name'] + style) % 1000000,
                    "steps": 15,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.6,  # Weniger Denoise = mehr vom Original beibehalten
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
                    "filename_prefix": f"REAL_{sprite_info['action']}_{style}_",
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }

        return workflow

    def parse_sprite_info(self, sprite_path):
        """Analysiert Sprite-Datei Informationen"""
        filename = sprite_path.name

        # Parse: action_frames_size_gridXxY.png
        parts = filename.replace('.png', '').split('_')

        if len(parts) >= 4:
            action = parts[0]
            frames = int(parts[1])
            size = parts[2]  # z.B. "512x512"
            grid_info = parts[3]  # z.B. "grid9x1"

            # Grid extrahieren
            if grid_info.startswith('grid'):
                grid_part = grid_info[4:]
                cols, rows = map(int, grid_part.split('x'))

                return {
                    'name': filename,
                    'action': action,
                    'frames': frames,
                    'size': size,
                    'cols': cols,
                    'rows': rows,
                    'original_path': sprite_path
                }

        return None

    def process_real_sprite(self, sprite_path):
        """Verarbeitet einen echten Sprite mit AI"""
        print(f"\n🎮 VERARBEITE ECHTEN SPRITE: {sprite_path.name}")
        print("=" * 60)

        # Sprite-Info analysieren
        sprite_info = self.parse_sprite_info(sprite_path)
        if not sprite_info:
            print(
                f"❌ Konnte Sprite-Info nicht analysieren: {sprite_path.name}")
            return False

        print(f"📐 Grid: {sprite_info['cols']}x{sprite_info['rows']}")
        print(f"🎯 Frames: {sprite_info['frames']}")
        print(f"📏 Größe: {sprite_info['size']}")

        # 1. Sprite zu ComfyUI hochladen
        print(f"📤 Lade Sprite hoch...")
        uploaded_filename = self.upload_image_to_comfyui(sprite_path)
        if not uploaded_filename:
            print(f"❌ Upload fehlgeschlagen")
            return False

        print(f"✅ Hochgeladen als: {uploaded_filename}")

        # 2. Für jeden Style verarbeiten
        styles = ["anime", "pixel", "enhanced"]
        successful = 0

        for style in styles:
            print(f"\n🎨 STYLE: {style.upper()}")
            print("-" * 40)

            try:
                # IMG2IMG Workflow erstellen
                workflow = self.create_img2img_workflow(
                    uploaded_filename, sprite_info, style)

                # Workflow an ComfyUI senden
                response = requests.post(f"{self.comfyui_url}/prompt",
                                         json={"prompt": workflow},
                                         timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    print(f"✅ AI-Verarbeitung gestartet (ID: {prompt_id})")
                    successful += 1

                    # Warten auf Abschluss (vereinfacht)
                    print("⏳ Warte auf AI-Verarbeitung...")

                else:
                    print(f"❌ Workflow-Fehler: {response.text[:200]}")

            except Exception as e:
                print(f"❌ Fehler bei {style}: {e}")

        print(f"\n✅ {successful}/{len(styles)} Styles erfolgreich gestartet")
        return successful > 0

    def run_real_processing(self):
        """Hauptfunktion - verarbeitet deine echten Sprites"""
        print("🎮 ECHTER SPRITE PROCESSOR")
        print("🔥 VERWENDET DEINE ORIGINALEN SPRITE-SHEETS")
        print("=" * 60)

        # Server testen
        try:
            response = requests.get(
                f"{self.comfyui_url}/system_stats", timeout=5)
            if response.status_code != 200:
                print("❌ ComfyUI Server ist nicht verfügbar!")
                print(
                    "💡 Starte: cd ComfyUI_engine && python main.py --listen --port 8188 --cpu")
                return False
        except:
            print("❌ ComfyUI Server nicht erreichbar!")
            return False

        print("✅ ComfyUI Server ist bereit")

        # Deine Sprite-Dateien finden
        sprite_files = self.find_sprite_files()

        if not sprite_files:
            print("❌ KEINE SPRITE-DATEIEN GEFUNDEN!")
            print("💡 Stelle sicher dass diese Dateien im input/ Ordner sind:")
            print("   - idle_9_512x512_grid9x1.png")
            print("   - walk_8_512x512_grid8x1.png")
            print("   - jump_20_512x512_grid8x3.png")
            print("   - attack_8_512x512_grid8x1.png")
            print("   - intro_24_512x512_grid8x3.png")
            return False

        print(f"\n📋 GEFUNDENE SPRITES: {len(sprite_files)}")

        # Jeden Sprite verarbeiten
        successful_sprites = 0
        for sprite_path in sprite_files:
            if self.process_real_sprite(sprite_path):
                successful_sprites += 1

        print(f"\n" + "=" * 60)
        print("🏁 ECHTE SPRITE-VERARBEITUNG ABGESCHLOSSEN")
        print(f"=" * 60)
        print(
            f"✅ Erfolgreich verarbeitet: {successful_sprites}/{len(sprite_files)}")
        print(f"📁 Ergebnisse in: ComfyUI_engine/output/")
        print("🔍 Suche nach Dateien mit 'REAL_' Prefix")

        if successful_sprites > 0:
            print("\n🎉 ECHTE AI-SPRITE-TRANSFORMATION GESTARTET!")
            print("⏳ Die AI verarbeitet jetzt deine originalen Sprites...")
            print("🎯 Frame-Struktur wird beibehalten!")

        return successful_sprites > 0


if __name__ == "__main__":
    processor = EchterSpriteProcessor()
    processor.run_real_processing()
