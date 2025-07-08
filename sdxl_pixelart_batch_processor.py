#!/usr/bin/env python3
"""
🎨 SDXL + PIXEL ART ENHANCED BATCH PROCESSOR
============================================
Kombiniert den ursprünglichen Pixel-Art Workflow mit SDXL + Pixel-Art LoRA
für verbesserte und konsistentere Pixel-Art Ergebnisse
"""

import json
import os
import time
import urllib.parse
from pathlib import Path
import requests
import base64
from PIL import Image
import numpy as np


class SDXLPixelArtBatchProcessor:
    def __init__(self, comfyui_server="127.0.0.1:8188"):
        self.server = comfyui_server
        self.base_url = f"http://{comfyui_server}"
        self.workflow_template = self.load_workflow_template()

        # Ausgabe-Verzeichnis
        self.output_dir = Path("output/sdxl_pixelart_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_workflow_template(self):
        """Lädt das Workflow-Template"""
        with open("sdxl_pixelart_combined_workflow.json", "r") as f:
            return json.load(f)

    def queue_prompt(self, prompt):
        """Sendet einen Workflow an ComfyUI"""
        p = {"prompt": prompt}
        data = json.dumps(p).encode('utf-8')
        req = requests.post(f"{self.base_url}/prompt", data=data)
        return json.loads(req.text)

    def get_history(self, prompt_id):
        """Holt den Verlauf eines Prompts"""
        req = requests.get(f"{self.base_url}/history/{prompt_id}")
        return json.loads(req.text)

    def get_images(self, ws, prompt):
        """Wartet auf und lädt die generierten Bilder herunter"""
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Ausführung abgeschlossen
            else:
                continue

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                output_images[node_id] = []
                for image in node_output['images']:
                    image_data = self.get_image(
                        image['filename'], image['subfolder'], image['type'])
                    output_images[node_id].append(image_data)

        return output_images

    def get_image(self, filename, subfolder, folder_type):
        """Lädt ein einzelnes Bild herunter"""
        data = {"filename": filename,
                "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        req = requests.get(f"{self.base_url}/view?{url_values}")
        return req.content

    def create_workflow_for_gif(self, gif_path, style_prompt="", seed=None):
        """Erstellt einen Workflow für eine spezifische GIF-Datei"""
        workflow = self.workflow_template.copy()

        # Setze Input-Datei
        filename = os.path.basename(gif_path)
        workflow["1"]["inputs"]["image"] = filename

        # Angepasster Prompt basierend auf Dateiname/Stil
        base_prompt = "pixel art, 8bit style, retro game graphics, limited color palette, sharp edges"
        if style_prompt:
            full_prompt = f"{base_prompt}, {style_prompt}"
        else:
            full_prompt = base_prompt

        workflow["4"]["inputs"]["text"] = full_prompt

        # Zufälliger Seed wenn nicht angegeben
        if seed is None:
            seed = int(time.time()) % 1000000
        workflow["9"]["inputs"]["seed"] = seed

        # Output-Dateiname
        base_name = Path(gif_path).stem
        workflow["11"]["inputs"]["filename_prefix"] = f"sdxl_enhanced_{base_name}"

        return workflow

    def process_gif_list(self, gif_files, style_prompts=None):
        """Verarbeitet eine Liste von GIF-Dateien"""
        print("🎨 SDXL + PIXEL ART ENHANCED BATCH PROCESSOR")
        print("=" * 60)
        print(f"📁 {len(gif_files)} GIF-Dateien zu verarbeiten")
        print(f"📂 Output: {self.output_dir}")
        print(f"🔗 ComfyUI Server: {self.server}")
        print("=" * 60)

        successful = 0
        failed = 0

        for i, gif_file in enumerate(gif_files, 1):
            try:
                print(
                    f"\n🔄 [{i:3d}/{len(gif_files)}] {os.path.basename(gif_file)}")

                # Style-Prompt für diese Datei
                style_prompt = ""
                if style_prompts and i-1 < len(style_prompts):
                    style_prompt = style_prompts[i-1]

                # Erstelle Workflow
                workflow = self.create_workflow_for_gif(gif_file, style_prompt)

                # Verarbeite in ComfyUI
                print("   📤 Sende an ComfyUI...")
                result = self.queue_prompt(workflow)

                if 'prompt_id' in result:
                    print(
                        f"   ✅ Workflow gesendet (ID: {result['prompt_id']})")
                    successful += 1
                else:
                    print(f"   ❌ Fehler beim Senden")
                    failed += 1

                # Kurze Pause zwischen Requests
                time.sleep(2)

            except Exception as e:
                print(f"   ❌ Fehler: {str(e)[:60]}...")
                failed += 1

        print(f"\n" + "=" * 60)
        print("🎯 BATCH PROCESSING ABGESCHLOSSEN")
        print(f"✅ Erfolgreich gesendet: {successful}")
        print(f"❌ Fehlgeschlagen: {failed}")
        print(f"📁 Output wird in: {self.output_dir} gespeichert")
        print("=" * 60)

        return successful, failed

    def generate_style_variations(self, gif_file, variations=3):
        """Erstellt mehrere Stil-Variationen für eine GIF-Datei"""
        style_prompts = [
            "vibrant colors, game boy color style, nostalgic",
            "muted earth tones, classic arcade aesthetic",
            "neon cyberpunk palette, futuristic retro style",
            "monochrome gameboy style, high contrast",
            "warm sunset colors, cozy indie game feel"
        ]

        results = []
        for i in range(min(variations, len(style_prompts))):
            workflow = self.create_workflow_for_gif(
                gif_file, style_prompts[i], seed=i*1000)
            results.append(workflow)

        return results


def main():
    """Beispiel-Verwendung"""

    # Liste der zu verarbeitenden GIF-Dateien
    gif_files = [
        "input/0e35f5b16b8ba60a10fdd360de075def_fast_transparent_converted.gif",
        "input/spinning_vinyl_clean_fast_transparent_converted.gif",
        "input/gym-roshi_2_fast_transparent_converted.gif",
        "input/rick-and-morty-fortnite_fast_transparent_converted.gif",
        # Weitere Dateien hinzufügen...
    ]

    # Optionale Style-Prompts für jede Datei
    style_prompts = [
        "character sprite, detailed animation",
        "music visualization, rhythmic patterns",
        "anime character, dynamic action",
        "cartoon style, vibrant colors",
    ]

    # Prozessor initialisieren
    processor = SDXLPixelArtBatchProcessor()

    # Verarbeitung starten
    success, failed = processor.process_gif_list(gif_files, style_prompts)

    print(f"\n🎉 Batch-Verarbeitung abgeschlossen!")
    print(f"Erfolg: {success}, Fehler: {failed}")


if __name__ == "__main__":
    main()
