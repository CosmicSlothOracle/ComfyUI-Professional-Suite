#!/usr/bin/env python3
"""
Automatischer Model-Downloader für ComfyUI Sprite-Workflow
==========================================================
Lädt alle notwendigen Modelle automatisch herunter.
"""

import os
import requests
from pathlib import Path
import json


def download_file(url, destination, description=""):
    """Lade Datei mit Progress-Bar herunter"""
    try:
        print(f"\n📥 {description}")
        print(f"   Ziel: {destination}")

        if destination.exists():
            print(f"   ✅ Bereits vorhanden - überspringe")
            return True

        destination.parent.mkdir(parents=True, exist_ok=True)

        print(f"   🔄 Starte Download...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end="")

        print(f"\r   ✅ Download erfolgreich")
        return True

    except Exception as e:
        print(f"\r   ❌ Fehler beim Download: {e}")
        if destination.exists():
            destination.unlink()
        return False


def main():
    print("🎯 ComfyUI Sprite-Workflow Model Downloader")
    print("=" * 50)

    # Basis-Verzeichnis
    base_dir = Path(".")
    models_dir = base_dir / "models"

    # Modell-URLs (öffentlich verfügbare)
    models = {
        # VAE Models (kleinere, öffentlich verfügbare)
        "vae": {
            "vae-ft-mse-840000-ema-pruned.ckpt": {
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
                "description": "Standard VAE für Stable Diffusion"
            }
        },

        # ControlNet Models
        "controlnet": {
            "control_v11p_sd15_openpose_fp16.safetensors": {
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
                "description": "OpenPose ControlNet - Pose-Erkennung"
            }
        }
    }

    print("🚀 Starte Download verfügbarer Modelle...")
    success_count = 0
    total_count = 0

    for category, files in models.items():
        print(f"\n📂 Lade {category.upper()} Modelle...")

        for filename, info in files.items():
            total_count += 1
            destination = models_dir / category / filename

            if download_file(info["url"], destination, info["description"]):
                success_count += 1

    print(f"\n" + "=" * 50)
    print(f"🎉 DOWNLOAD ABGESCHLOSSEN!")
    print(f"✅ Erfolgreich: {success_count}/{total_count}")
    if success_count < total_count:
        print(f"❌ Fehlgeschlagen: {total_count - success_count}")

    # Erstelle Anleitung für manuelle Downloads
    print(f"\n📋 Manuelle Downloads erforderlich:")
    print("=" * 50)
    print("Für die besten Ergebnisse benötigst du noch:")
    print()
    print("1. 🤖 Basis-Checkpoints (von civitai.com):")
    print("   - DreamShaper XL Turbo v2.1")
    print("   - RealVisXL v4.0")
    print("   → Speichere in: models/checkpoints/")
    print()
    print("2. 🎨 LoRA-Models (von civitai.com):")
    print("   - Pixel Art XL v1.5")
    print("   - Anime Style XL")
    print("   → Speichere in: models/loras/")
    print()
    print("3. 📋 Weitere ControlNet-Models (von huggingface.co):")
    print("   - control_v11f1p_sd15_depth_fp16.safetensors")
    print("   - control_v11p_sd15_lineart_fp16.safetensors")
    print("   → Speichere in: models/controlnet/")
    print()
    print("💡 Tipp: Starte ComfyUI und verwende den ComfyUI Manager")
    print("   für einfacheren Model-Download!")
    print("=" * 50)


if __name__ == "__main__":
    main()
