#!/usr/bin/env python3
"""
VAPORWAVE VIDEO PROCESSOR
Führt den ComfyUI Workflow für die Vaporwave-Konvertierung aus
"""

import os
import sys
import json
import time
import requests
from pathlib import Path


def check_server():
    """Prüfe ComfyUI Server"""
    try:
        response = requests.get(
            "http://127.0.0.1:8188/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_workflow():
    """Führe Vaporwave Workflow aus"""

    if not check_server():
        print("❌ ComfyUI Server nicht erreichbar!")
        print("Starte zuerst: python main.py")
        return False

    print("✅ ComfyUI Server bereit")

    # Lade Workflow
    workflow_path = Path("workflows/video_to_vaporwave_gifs_workflow.json")

    if not workflow_path.exists():
        print(f"❌ Workflow nicht gefunden: {workflow_path}")
        return False

    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # Erstelle Output-Ordner
    Path("output/vaporwave_gifs").mkdir(parents=True, exist_ok=True)
    Path("output/vaporwave_frames").mkdir(parents=True, exist_ok=True)

    print("🚀 Starte Vaporwave-Verarbeitung...")

    # Führe Workflow aus
    try:
        data = {"prompt": workflow}
        response = requests.post("http://127.0.0.1:8188/prompt", json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workflow gestartet!")
            print(
                f"🎬 Verarbeitung läuft... (Prompt ID: {result.get('prompt_id', 'unknown')})")
            print("📁 Output wird gespeichert in:")
            print("   • output/vaporwave_gifs/ (GIF-Dateien)")
            print("   • output/vaporwave_frames/ (Einzelbilder)")
            return True
        else:
            print(f"❌ Fehler: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Ausführungsfehler: {e}")
        return False


if __name__ == "__main__":
    print("🎬 VAPORWAVE VIDEO PROCESSOR")
    print("=" * 40)

    success = run_workflow()

    if success:
        print("\n🎉 Workflow erfolgreich gestartet!")
        print("⏳ Verarbeitung läuft im Hintergrund...")
        print("📱 Überprüfe die ComfyUI Web-UI für den Fortschritt")
    else:
        print("\n💥 Workflow konnte nicht gestartet werden!")
        sys.exit(1)
