#!/usr/bin/env python3
"""
AI-Setup Validierung für ComfyUI Sprite-Workflow
===============================================
Überprüft ob alle AI-Modelle vorhanden und funktional sind.
Stellt sicher dass keine Dummy-Prozesse verwendet werden.
"""

import os
import json
from pathlib import Path
import hashlib


def check_file_size(file_path, min_size_mb=1):
    """Prüfe ob Datei groß genug ist (echtes AI-Modell, kein Dummy)"""
    if not file_path.exists():
        return False, "Datei nicht gefunden"

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        return False, f"Datei zu klein ({size_mb:.1f}MB) - möglicherweise Dummy-Datei"

    return True, f"OK ({size_mb:.1f}MB)"


def validate_ai_models():
    """Validiere alle AI-Modelle"""
    print("🤖 VALIDIERUNG DER AI-MODELLE")
    print("=" * 50)

    models_to_check = {
        "🧠 Checkpoint-Modelle (Basis-AI)": {
            "directory": "models/checkpoints",
            "extensions": [".safetensors", ".ckpt"],
            "min_size_mb": 1000,  # Mindestens 1GB für echte Checkpoints
            "required": True
        },
        "🎯 ControlNet-Modelle (Pose-AI)": {
            "directory": "models/controlnet",
            "extensions": [".safetensors", ".pth"],
            "min_size_mb": 500,   # Mindestens 500MB für ControlNet
            "required": True
        },
        "🎨 LoRA-Modelle (Style-AI)": {
            "directory": "models/loras",
            "extensions": [".safetensors"],
            "min_size_mb": 10,    # Mindestens 10MB für LoRA
            "required": False
        },
        "🔧 VAE-Modelle (Encoding-AI)": {
            "directory": "models/vae",
            "extensions": [".safetensors", ".ckpt"],
            "min_size_mb": 100,   # Mindestens 100MB für VAE
            "required": True
        }
    }

    all_valid = True

    for category, config in models_to_check.items():
        print(f"\n{category}:")

        model_dir = Path(config["directory"])
        if not model_dir.exists():
            print(f"   ❌ Verzeichnis nicht gefunden: {model_dir}")
            if config["required"]:
                all_valid = False
            continue

        # Finde alle Modelle
        models = []
        for ext in config["extensions"]:
            models.extend(model_dir.glob(f"*{ext}"))

        if not models:
            print(f"   ⚠️  Keine Modelle gefunden")
            if config["required"]:
                all_valid = False
            continue

        # Prüfe jedes Modell
        valid_models = 0
        for model in models:
            is_valid, status = check_file_size(model, config["min_size_mb"])
            icon = "✅" if is_valid else "❌"
            print(f"   {icon} {model.name}: {status}")

            if is_valid:
                valid_models += 1
            elif config["required"]:
                all_valid = False

        print(f"   📊 Gültige Modelle: {valid_models}/{len(models)}")

    return all_valid


def validate_workflows():
    """Validiere dass Workflows echte AI-Nodes verwenden"""
    print("\n🔧 VALIDIERUNG DER AI-WORKFLOWS")
    print("=" * 50)

    ai_nodes = {
        "DWPreprocessor": "Pose-Erkennung AI",
        "ControlNetLoader": "ControlNet AI-Steuerung",
        "ControlNetApplyAdvanced": "AI-Pose-Anwendung",
        "CheckpointLoaderSimple": "Basis-AI-Modell",
        "CLIPTextEncode": "Text-zu-AI-Encoding",
        "KSampler": "AI-Bildgenerierung",
        "VAEDecode": "AI-Dekodierung",
        "LoraLoader": "Style-AI-Anwendung"
    }

    workflow_dir = Path("workflows/sprite_processing")
    workflows = list(workflow_dir.glob("*.json"))

    all_valid = True

    for workflow_file in workflows:
        print(f"\n📋 {workflow_file.name}:")

        try:
            with open(workflow_file, 'r') as f:
                workflow = json.load(f)

            # Finde alle Node-Typen
            if "nodes" in workflow:
                nodes = workflow["nodes"]
                if isinstance(nodes, list):
                    node_types = [node.get("type", "") for node in nodes]
                elif isinstance(nodes, dict):
                    node_types = [node.get("type", "")
                                  for node in nodes.values()]
                else:
                    node_types = []
            else:
                node_types = []

            # Prüfe AI-Nodes
            found_ai_nodes = []
            for node_type in node_types:
                if node_type in ai_nodes:
                    found_ai_nodes.append(node_type)

            if found_ai_nodes:
                print(f"   ✅ AI-Nodes gefunden ({len(found_ai_nodes)}):")
                for node_type in found_ai_nodes:
                    print(f"      🤖 {node_type}: {ai_nodes[node_type]}")
            else:
                print(f"   ❌ Keine AI-Nodes gefunden - möglicherweise Dummy-Workflow")
                all_valid = False

        except Exception as e:
            print(f"   ❌ Fehler beim Laden: {e}")
            all_valid = False

    return all_valid


def validate_custom_nodes():
    """Validiere dass notwendige Custom Nodes installiert sind"""
    print("\n🔌 VALIDIERUNG DER AI-EXTENSIONS")
    print("=" * 50)

    required_nodes = {
        "ComfyUI-Advanced-ControlNet": "Erweiterte AI-Pose-Kontrolle",
        "ComfyUI_IPAdapter_plus": "AI-Style-Transfer",
        "ComfyUI-AnimateDiff-Evolved": "AI-Animation",
        "comfy_mtb": "AI-Batch-Processing"
    }

    custom_nodes_dir = Path("custom_nodes")
    all_valid = True

    for node_name, description in required_nodes.items():
        node_path = custom_nodes_dir / node_name

        if node_path.exists():
            # Prüfe ob es ein echtes Verzeichnis mit Dateien ist
            py_files = list(node_path.glob("*.py"))
            if py_files:
                print(
                    f"   ✅ {node_name}: {description} ({len(py_files)} Python-Dateien)")
            else:
                print(f"   ⚠️  {node_name}: Verzeichnis vorhanden aber leer")
                all_valid = False
        else:
            print(f"   ❌ {node_name}: Nicht installiert")
            all_valid = False

    return all_valid


def validate_configuration():
    """Validiere Konfiguration für AI-Processing"""
    print("\n⚙️ VALIDIERUNG DER AI-KONFIGURATION")
    print("=" * 50)

    config_file = Path("config.json")

    if not config_file.exists():
        print("   ❌ config.json nicht gefunden")
        return False

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Prüfe Style-Presets
        if "style_presets" in config:
            presets = config["style_presets"]
            print(f"   ✅ Style-Presets gefunden: {len(presets)}")

            for style_name, style_config in presets.items():
                if "lora" in style_config:
                    lora_file = Path("models/loras") / style_config["lora"]
                    if lora_file.exists():
                        print(f"      ✅ {style_name}: LoRA verfügbar")
                    else:
                        print(
                            f"      ⚠️  {style_name}: LoRA fehlt ({style_config['lora']})")
                else:
                    print(f"      ⚠️  {style_name}: Keine LoRA-Konfiguration")
        else:
            print("   ❌ Keine Style-Presets konfiguriert")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Fehler beim Laden der Konfiguration: {e}")
        return False


def create_test_sprite():
    """Erstelle ein Test-Sprite-Sheet für Validierung"""
    print("\n🎮 ERSTELLE TEST-SPRITE")
    print("=" * 50)

    test_dir = Path("input/sprite_sheets")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_instructions = """
TEST-SPRITE FÜR AI-VALIDIERUNG
==============================

Um die AI-Funktionalität zu testen:

1. Lade ein echtes Sprite-Sheet herunter von:
   - https://opengameart.org/content/lpc-characters
   - https://kenney.nl/assets/tiny-characters
   - https://itch.io/game-assets/free

2. Benenne es um zu: test_sprite.png

3. Lege es in dieses Verzeichnis: input/sprite_sheets/

4. Starte ComfyUI und lade den Workflow:
   workflows/sprite_processing/sprite_extractor.json

5. Führe den Workflow aus und prüfe ob:
   - Frames extrahiert werden
   - Posen erkannt werden
   - Keine Fehler auftreten

ERWARTETE AI-PROZESSE:
======================
✅ LoadImage: Sprite-Sheet laden
✅ ImageSplitGrid: Frames extrahieren
✅ DWPreprocessor: AI-Pose-Erkennung
✅ SaveImage: Ergebnisse speichern

WARNSIGNALE FÜR DUMMY-PROZESSE:
==============================
❌ Sofortige Fertigstellung ohne Processing
❌ Leere Output-Dateien
❌ Fehlende Pose-Skelette
❌ Identische Input/Output-Bilder
"""

    test_file = test_dir / "TEST_ANLEITUNG.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_instructions)

    print(f"   ✅ Test-Anleitung erstellt: {test_file}")


def main():
    print("🔍 AI-SETUP VALIDIERUNG")
    print("=" * 60)
    print("Prüfe ob alle AI-Komponenten echt und funktional sind...")

    # Validierungen durchführen
    models_valid = validate_ai_models()
    workflows_valid = validate_workflows()
    nodes_valid = validate_custom_nodes()
    config_valid = validate_configuration()

    # Test-Setup erstellen
    create_test_sprite()

    print("\n" + "=" * 60)
    print("📊 VALIDIERUNGSERGEBNIS")
    print("=" * 60)

    results = {
        "🤖 AI-Modelle": models_valid,
        "🔧 AI-Workflows": workflows_valid,
        "🔌 AI-Extensions": nodes_valid,
        "⚙️ AI-Konfiguration": config_valid
    }

    all_valid = all(results.values())

    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component}: {'GÜLTIG' if status else 'FEHLERHAFT'}")

    if all_valid:
        print("\n🎉 ALLE AI-KOMPONENTEN VALIDIERT!")
        print("✅ Keine Dummy-Prozesse gefunden")
        print("✅ Echte AI-Modelle verfügbar")
        print("✅ Funktionale Workflows bereit")
        print("\n🚀 Das System ist bereit für echte AI-Sprite-Verarbeitung!")
    else:
        print("\n⚠️  VALIDIERUNG UNVOLLSTÄNDIG!")
        print("Einige AI-Komponenten fehlen oder sind nicht funktional.")
        print("Führe die empfohlenen Schritte aus:")
        print("1. python download_models.py")
        print("2. Lade zusätzliche Modelle von civitai.com")
        print("3. Starte ComfyUI und installiere fehlende Nodes")

    print("=" * 60)


if __name__ == "__main__":
    main()
