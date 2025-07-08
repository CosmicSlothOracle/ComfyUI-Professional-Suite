#!/usr/bin/env python3
"""
Test-Script für den Sprite-Workflow
===================================
Demonstriert die Verwendung des automatisierten Sprite-Processing-Workflows.
"""

import os
import json
from pathlib import Path


def check_setup():
    """Prüfe ob alle notwendigen Komponenten vorhanden sind"""
    print("🔍 Überprüfe Setup...")

    checks = {
        "Workflow-Dateien": Path("workflows/sprite_processing").exists(),
        "Input-Verzeichnis": Path("input/sprite_sheets").exists(),
        "Output-Verzeichnis": Path("output").exists(),
        "Models-Verzeichnis": Path("models").exists(),
        "Custom Nodes": Path("custom_nodes").exists(),
        "Config-Datei": Path("config.json").exists(),
        "Startup-Script": Path("start_sprite_workflow.bat").exists()
    }

    all_good = True
    for component, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}")
        if not status:
            all_good = False

    return all_good


def check_models():
    """Prüfe verfügbare Modelle"""
    print("\n📦 Verfügbare Modelle:")

    model_dirs = {
        "Checkpoints": "models/checkpoints",
        "ControlNet": "models/controlnet",
        "LoRAs": "models/loras",
        "VAE": "models/vae"
    }

    for category, path in model_dirs.items():
        model_path = Path(path)
        if model_path.exists():
            models = list(model_path.glob("*.safetensors")) + \
                list(model_path.glob("*.ckpt")) + \
                list(model_path.glob("*.pth"))
            print(f"   📂 {category}: {len(models)} Modelle")
            for model in models[:3]:  # Zeige nur die ersten 3
                print(f"      - {model.name}")
            if len(models) > 3:
                print(f"      ... und {len(models) - 3} weitere")
        else:
            print(f"   📂 {category}: Verzeichnis nicht gefunden")


def show_workflow_info():
    """Zeige Workflow-Informationen"""
    print("\n📋 Verfügbare Workflows:")

    workflow_dir = Path("workflows/sprite_processing")
    if workflow_dir.exists():
        workflows = list(workflow_dir.glob("*.json"))
        for workflow in workflows:
            print(f"   🎭 {workflow.name}")
            print(f"      Pfad: {workflow}")
    else:
        print("   ❌ Workflow-Verzeichnis nicht gefunden")


def create_example_sprite():
    """Erstelle ein Beispiel-Sprite-Sheet Text-Datei mit Anweisungen"""
    print("\n📝 Erstelle Beispiel-Anweisungen...")

    example_dir = Path("input/sprite_sheets")
    example_dir.mkdir(parents=True, exist_ok=True)

    instructions_file = example_dir / "ANLEITUNG.txt"

    instructions = """
🎮 SPRITE-SHEET ANLEITUNG
========================

So verwendest du den Sprite-Workflow:

1. SPRITE-SHEET VORBEREITEN:
   - Lege deine Sprite-Sheets (PNG/JPG/GIF) in dieses Verzeichnis
   - Empfohlene Auflösung: Mindestens 256x256 für beste Pose-Erkennung
   - Frame-Größe sollte einheitlich sein (z.B. 64x64, 128x128)

2. BEISPIEL SPRITE-SHEET STRUKTUR:
   [Frame1] [Frame2] [Frame3] [Frame4]
   [Frame5] [Frame6] [Frame7] [Frame8]

   - Frames sind in einem Raster angeordnet
   - Jeder Frame zeigt eine Pose/Animation

3. WORKFLOW STARTEN:
   - Starte ComfyUI mit: start_sprite_workflow.bat
   - Lade Workflow: workflows/sprite_processing/sprite_extractor.json
   - Passe Frame-Größe an (Standard: 64x64)
   - Führe Queue aus

4. STYLE-TRANSFER:
   - Lade Workflow: workflows/sprite_processing/style_transfer.json
   - Wähle gewünschten Style (Anime/Pixel/Realistic)
   - Verwende extrahierte Pose-Daten als Referenz

5. VERFÜGBARE STYLES:
   - Anime: Vibrant, cel-shaded, manga-style
   - Pixel Art: 8-bit, retro, blocky design
   - Realistic: Photorealistic, detailed textures

BEISPIEL-DATEIEN ZUM TESTEN:
============================
Du kannst diese kostenlose Sprite-Sheets verwenden:

- OpenGameArt.org
- Kenney.nl (Game Assets)
- LPC (Liberated Pixel Cup) Assets
- Itch.io (Free Game Assets)

EMPFOHLENE TEST-SPRITES:
========================
- Character Walk Cycles (8-12 Frames)
- Fighting Game Moves (4-8 Frames)
- Idle Animations (2-4 Frames)
- Attack Animations (6-10 Frames)

TROUBLESHOOTING:
================
- Pose wird nicht erkannt: Frame-Größe erhöhen
- Schlechte Qualität: Eingabe-Auflösung verbessern
- Langsame Performance: Batch-Size reduzieren
- Model-Fehler: ComfyUI Manager → Install Missing Nodes

Viel Erfolg mit deinem Sprite-Processing! 🎨
"""

    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)

    print(f"   ✅ Anleitung erstellt: {instructions_file}")


def show_next_steps():
    """Zeige nächste Schritte"""
    print("\n🚀 NÄCHSTE SCHRITTE:")
    print("=" * 50)
    print("1. 📥 Modelle herunterladen:")
    print("   - Führe aus: python download_models.py")
    print("   - Oder lade manuell von civitai.com herunter")
    print()
    print("2. 🎮 ComfyUI starten:")
    print("   - Doppelklick: start_sprite_workflow.bat")
    print("   - Oder: python main.py --listen --port 8188")
    print()
    print("3. 📋 Workflows laden:")
    print("   - workflows/sprite_processing/sprite_extractor.json")
    print("   - workflows/sprite_processing/style_transfer.json")
    print()
    print("4. 🎨 Erste Tests:")
    print("   - Lege Test-Sprites in: input/sprite_sheets/")
    print("   - Folge der Anleitung in: input/sprite_sheets/ANLEITUNG.txt")
    print()
    print("5. 🛠️ Beim ersten Start:")
    print("   - ComfyUI Manager → Install Missing Nodes")
    print("   - Models automatisch herunterladen lassen")
    print()
    print("💡 Tipp: Lese README_SPRITE_WORKFLOW.md für detaillierte Anleitung!")


def main():
    print("🎯 ComfyUI Sprite-Workflow Test & Setup-Prüfung")
    print("=" * 60)

    # Setup prüfen
    if check_setup():
        print("\n✅ Setup-Prüfung erfolgreich!")
    else:
        print("\n❌ Setup unvollständig - führe install_sprite_workflow.bat aus")
        return

    # Modelle prüfen
    check_models()

    # Workflow-Info
    show_workflow_info()

    # Beispiel-Anweisungen erstellen
    create_example_sprite()

    # Nächste Schritte
    show_next_steps()

    print("\n" + "=" * 60)
    print("🎉 SPRITE-WORKFLOW BEREIT ZUM EINSATZ!")
    print("=" * 60)


if __name__ == "__main__":
    main()
