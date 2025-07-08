#!/usr/bin/env python3
"""
Einfacher Sprite-Test mit grundlegenden ComfyUI Funktionen
"""

import json
import os
import sys
import time
import requests
import subprocess
from pathlib import Path


def create_simple_workflow():
    """Erstellt einen einfachen Workflow für Sprite-Verarbeitung"""
    workflow = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "sdxl.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": "anime character, idle pose, pixel art style, 2d game sprite",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Prompt)"
            }
        },
        "7": {
            "inputs": {
                "text": "blurry, low quality, bad anatomy",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Negative)"
            }
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "sprite_test",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        }
    }
    return workflow


def test_basic_generation():
    """Testet grundlegende Bildgenerierung"""
    print("🎮 TESTE EINFACHE SPRITE-GENERIERUNG")
    print("=" * 50)

    # ComfyUI Server URL
    url = "http://127.0.0.1:8188"

    # Server-Verfügbarkeit prüfen
    try:
        response = requests.get(f"{url}/system_stats", timeout=5)
        if response.status_code != 200:
            print("❌ ComfyUI Server ist nicht verfügbar")
            return False
    except:
        print("❌ ComfyUI Server ist nicht erreichbar")
        return False

    print("✅ ComfyUI Server ist bereit")

    # Einfachen Workflow erstellen
    workflow = create_simple_workflow()

    # Workflow senden
    try:
        response = requests.post(f"{url}/prompt", json={"prompt": workflow})
        if response.status_code == 200:
            result = response.json()
            prompt_id = result.get("prompt_id")
            print(f"✅ Workflow gesendet (ID: {prompt_id})")

            # Auf Ergebnis warten
            print("⏳ Warte auf Generierung...")
            max_attempts = 60
            for attempt in range(max_attempts):
                try:
                    history_response = requests.get(
                        f"{url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            if status.get("completed", False):
                                print("✅ Generierung abgeschlossen!")
                                outputs = history[prompt_id].get("outputs", {})
                                if outputs:
                                    print(f"📁 Ergebnisse: {outputs}")
                                return True
                            elif "error" in status:
                                print(f"❌ Fehler: {status['error']}")
                                return False
                except Exception as e:
                    print(f"⚠️ Fehler beim Status-Check: {e}")

                time.sleep(2)

            print("⏰ Timeout erreicht")
            return False
        else:
            print(f"❌ Fehler beim Senden: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        return False


def start_server_and_test():
    """Startet Server und führt Test durch"""
    print("🚀 Starte ComfyUI Server...")

    # Server starten
    server_process = subprocess.Popen([
        sys.executable, "main.py",
        "--listen", "--port", "8188", "--cpu"
    ], cwd="ComfyUI_engine", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Warten bis Server bereit
        print("⏳ Warte auf Server...")
        time.sleep(10)  # Server-Startzeit abwarten

        # Test durchführen
        success = test_basic_generation()

        if success:
            print("\n✅ SPRITE-TEST ERFOLGREICH!")
            print("🎯 Das System ist bereit für komplexere Workflows")
        else:
            print("\n❌ SPRITE-TEST FEHLGESCHLAGEN")
            print("🔧 Überprüfe ComfyUI Installation und Modelle")

        return success

    finally:
        # Server stoppen
        server_process.terminate()
        server_process.wait()
        print("🛑 Server gestoppt")


if __name__ == "__main__":
    try:
        start_server_and_test()
    except KeyboardInterrupt:
        print("\n⏹️ Test abgebrochen")
    except Exception as e:
        print(f"\n💥 Fehler: {e}")
