#!/usr/bin/env python3
"""
Automatisierte Verarbeitung aller Sprite-Sheets
==============================================
Wendet AI-Workflows auf alle vorhandenen Sprite-Sheets an.
"""

import os
import json
import shutil
from pathlib import Path
import subprocess
import time

# Sprite-Sheet Konfigurationen basierend auf deinen Dateien
SPRITE_CONFIGS = {
    "idle_9_512x512_grid9x1.png": {
        "grid": [9, 1],
        "frame_count": 9,
        "animation_type": "idle",
        "description": "Idle Animation - 9 Frames"
    },
    "intro_24_512x512_grid8x3.png": {
        "grid": [8, 3],
        "frame_count": 24,
        "animation_type": "intro",
        "description": "Intro Sequence - 24 Frames"
    },
    "jump_20_512x512_grid8x3.png": {
        "grid": [8, 3],
        "frame_count": 20,
        "animation_type": "jump",
        "description": "Jump Animation - 20 Frames"
    },
    "walk_8_512x512_grid8x1.png": {
        "grid": [8, 1],
        "frame_count": 8,
        "animation_type": "walk",
        "description": "Walk Cycle - 8 Frames"
    },
    "attack_8_512x512_grid8x1.png": {
        "grid": [8, 1],
        "frame_count": 8,
        "animation_type": "attack",
        "description": "Attack Animation - 8 Frames"
    }
}

# Style-Presets für verschiedene Outputs
STYLE_PRESETS = {
    "anime": {
        "positive": "high quality anime character art, detailed clothing, vibrant colors, masterpiece, best quality, cel shaded, clean lines, dynamic pose",
        "negative": "blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality",
        "lora": "pixel-art-xl-lora.safetensors",
        "lora_strength": 0.8
    },
    "pixel_art": {
        "positive": "pixel art character, 8bit style, retro gaming, clean pixels, detailed sprite art, game character",
        "negative": "blurry, anti-aliased, smooth, modern art, realistic, photographic",
        "lora": "pixel-art-xl-lora.safetensors",
        "lora_strength": 1.2
    },
    "enhanced": {
        "positive": "high resolution character art, detailed textures, enhanced quality, professional artwork, clean design",
        "negative": "low quality, blurry, pixelated, artifacts, distorted",
        "lora": "",
        "lora_strength": 0.0
    }
}


def create_workflow_for_sprite(sprite_file, config, style="anime"):
    """Erstelle einen spezifischen Workflow für ein Sprite-Sheet"""

    # Basis-Workflow laden
    with open("workflows/sprite_processing/batch_sprite_processor.json", 'r') as f:
        workflow = json.load(f)

    # Anpasse LoadImage Node
    for node in workflow["nodes"]:
        if node["type"] == "LoadImage":
            node["widgets_values"][0] = sprite_file
            node["title"] = f"Input: {sprite_file}"

        # Anpasse ImageSplitGrid Node
        elif node["type"] == "ImageSplitGrid":
            node["widgets_values"] = config["grid"]
            node["title"] = f"Split {config['grid'][0]}x{config['grid'][1]} Grid"

        # Anpasse Prompts basierend auf Animation
        elif node["type"] == "CLIPTextEncode" and "positive" in node.get("title", ""):
            animation_context = {
                "idle": "standing idle pose, calm expression, relaxed stance",
                "walk": "dynamic walking pose, movement, stepping forward",
                "attack": "action pose, combat stance, attacking motion",
                "jump": "jumping pose, airborne, athletic movement",
                "intro": "dramatic pose, introduction scene, character reveal"
            }

            base_prompt = STYLE_PRESETS[style]["positive"]
            anim_prompt = animation_context.get(config["animation_type"], "")
            node["widgets_values"][0] = f"{base_prompt}, {anim_prompt}"

        elif node["type"] == "CLIPTextEncode" and "negative" in node.get("title", ""):
            node["widgets_values"][0] = STYLE_PRESETS[style]["negative"]

        # Anpasse Batch-Größe für Frame-Count
        elif node["type"] == "EmptyLatentImage":
            node["widgets_values"][2] = config["frame_count"]  # Batch size

        # Anpasse Output-Pfade
        elif node["type"] == "SaveImage":
            current_prefix = node["widgets_values"][0]
            animation_type = config["animation_type"]

            if "pose_detection" in current_prefix:
                node["widgets_values"][0] = f"pose_detection/{animation_type}_poses_"
            elif "extracted_frames" in current_prefix:
                node["widgets_values"][0] = f"extracted_frames/{animation_type}_frames_"
            elif "styled_sprites" in current_prefix:
                node["widgets_values"][0] = f"styled_sprites/{animation_type}_{style}_"

    return workflow


def setup_output_directories():
    """Erstelle Output-Verzeichnisse"""
    directories = [
        "output/pose_detection",
        "output/extracted_frames",
        "output/styled_sprites",
        "output/batch_results"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Verzeichnis erstellt: {directory}")


def process_sprite_sheet(sprite_file, config, style="anime"):
    """Verarbeite ein einzelnes Sprite-Sheet"""
    print(f"\n🎮 VERARBEITE: {sprite_file}")
    print(f"   📐 Grid: {config['grid'][0]}x{config['grid'][1]}")
    print(f"   🎯 Frames: {config['frame_count']}")
    print(f"   🎨 Style: {style}")
    print(f"   📝 {config['description']}")

    # Prüfe ob Datei existiert
    sprite_path = Path(f"input/sprite_sheets/{sprite_file}")
    if not sprite_path.exists():
        print(f"   ❌ Datei nicht gefunden: {sprite_path}")
        return False

    # Erstelle spezifischen Workflow
    workflow = create_workflow_for_sprite(sprite_file, config, style)

    # Speichere temporären Workflow
    temp_workflow = f"temp_workflow_{config['animation_type']}_{style}.json"
    with open(temp_workflow, 'w') as f:
        json.dump(workflow, f, indent=2)

    print(f"   ✅ Workflow erstellt: {temp_workflow}")

    # Hier würde normalerweise ComfyUI API aufgerufen werden
    # Für jetzt nur den Workflow speichern
    final_workflow = f"workflows/sprite_processing/{config['animation_type']}_{style}_workflow.json"
    shutil.copy(temp_workflow, final_workflow)

    # Cleanup
    os.remove(temp_workflow)

    print(f"   🎯 Workflow gespeichert: {final_workflow}")
    return True


def create_batch_summary():
    """Erstelle eine Zusammenfassung aller erstellten Workflows"""
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sprites": len(SPRITE_CONFIGS),
        "workflows_created": [],
        "instructions": {
            "manual_processing": [
                "1. Starte ComfyUI: python main.py --listen",
                "2. Öffne Browser: http://localhost:8188",
                "3. Lade gewünschten Workflow aus workflows/sprite_processing/",
                "4. Klicke 'Queue Prompt' um AI-Processing zu starten"
            ],
            "batch_processing": [
                "1. Verwende ComfyUI API für automatisierte Verarbeitung",
                "2. Alle Workflows sind vorkonfiguriert und bereit",
                "3. Output wird in entsprechende Unterverzeichnisse gespeichert"
            ]
        }
    }

    # Sammle alle erstellten Workflows
    workflow_dir = Path("workflows/sprite_processing")
    for workflow_file in workflow_dir.glob("*_workflow.json"):
        summary["workflows_created"].append(workflow_file.name)

    # Speichere Zusammenfassung
    with open("output/batch_results/processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("🎮 AUTOMATISIERTE SPRITE-SHEET VERARBEITUNG")
    print("=" * 60)

    # Setup
    setup_output_directories()

    print(f"\n📋 GEFUNDENE SPRITE-SHEETS: {len(SPRITE_CONFIGS)}")

    # Verarbeite alle Sprite-Sheets mit verschiedenen Styles
    processed_count = 0

    for sprite_file, config in SPRITE_CONFIGS.items():
        for style in STYLE_PRESETS.keys():
            success = process_sprite_sheet(sprite_file, config, style)
            if success:
                processed_count += 1

    # Erstelle Zusammenfassung
    summary = create_batch_summary()

    print("\n" + "=" * 60)
    print("📊 VERARBEITUNG ABGESCHLOSSEN")
    print("=" * 60)
    print(f"✅ Sprite-Sheets: {len(SPRITE_CONFIGS)}")
    print(f"✅ Styles pro Sheet: {len(STYLE_PRESETS)}")
    print(f"✅ Workflows erstellt: {len(summary['workflows_created'])}")

    print(f"\n📁 ERSTELLTE WORKFLOWS:")
    for workflow in summary['workflows_created']:
        print(f"   🔧 {workflow}")

    print(f"\n🚀 NÄCHSTE SCHRITTE:")
    print("1. Starte ComfyUI: python main.py --listen")
    print("2. Öffne: http://localhost:8188")
    print("3. Lade gewünschten Workflow aus workflows/sprite_processing/")
    print("4. Ändere bei Bedarf Parameter (Prompts, Styles, etc.)")
    print("5. Klicke 'Queue Prompt' für AI-Processing")

    print(f"\n📄 Detaillierte Anleitung: output/batch_results/processing_summary.json")


if __name__ == "__main__":
    main()
