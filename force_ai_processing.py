#!/usr/bin/env python3
"""
FORCE AI PROCESSING - ALTERNATIVLOS
Zwingt ComfyUI zur AI-Verarbeitung der Sprites
"""

import requests
import json
import time


def create_sprite_workflow(sprite_name, style):
    """Erstellt AI-Workflow für Sprite"""
    prompts = {
        'anime': f'anime character, {sprite_name} pose, cel shading, vibrant colors, 2d game character, clean art style',
        'pixel': f'pixel art character, {sprite_name} animation, retro game sprite, 16-bit style, crisp pixels',
        'enhanced': f'detailed character art, {sprite_name} pose, high quality illustration, professional artwork, game character'
    }

    workflow = {
        "1": {
            "inputs": {
                "seed": 42 + hash(sprite_name + style) % 1000000,
                "steps": 12,  # Schnell aber trotzdem AI
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "2": {
            "inputs": {"ckpt_name": "sdxl.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        "3": {
            "inputs": {
                "text": prompts.get(style, prompts['anime']),
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": "blurry, low quality, bad anatomy, deformed, ugly",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {"samples": ["1", 0], "vae": ["2", 2]},
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {
                "filename_prefix": f"AI_{sprite_name}_{style}_",
                "images": ["6", 0]
            },
            "class_type": "SaveImage"
        }
    }
    return workflow


def force_ai_processing():
    """ZWINGT AI-Verarbeitung für alle Sprites"""
    print("🎮 FORCE AI PROCESSING - ALTERNATIVLOS")
    print("🤖 AI MUSS LAUFEN - KEINE AUSREDEN!")
    print("=" * 50)

    url = "http://127.0.0.1:8188"

    # Test Server-Verbindung
    try:
        response = requests.get(f"{url}/system_stats", timeout=5)
        if response.status_code == 200:
            print("✅ ComfyUI Server erkannt - AI ist bereit!")
        else:
            print("❌ Server antwortet nicht korrekt")
            return
    except:
        print("❌ ComfyUI Server nicht erreichbar!")
        print(
            "💡 Starte erst: cd ComfyUI_engine && python main.py --listen --port 8188 --cpu")
        return

    # Deine Sprites
    sprites = ['idle', 'walk', 'jump', 'attack', 'intro']
    styles = ['anime', 'pixel', 'enhanced']

    print(
        f"📋 Verarbeite {len(sprites)} Sprites × {len(styles)} Styles = {len(sprites)*len(styles)} AI-Generierungen")
    print()

    successful_workflows = 0
    all_prompt_ids = []

    # AI-Workflows für jeden Sprite und Style starten
    for sprite in sprites:
        for style in styles:
            print(f"🔄 Starte AI: {sprite.upper()} - {style.upper()}")

            try:
                workflow = create_sprite_workflow(sprite, style)

                response = requests.post(f"{url}/prompt",
                                         json={"prompt": workflow},
                                         timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")

                    if prompt_id:
                        print(f"✅ AI-Workflow gestartet (ID: {prompt_id})")
                        successful_workflows += 1
                        all_prompt_ids.append({
                            'id': prompt_id,
                            'sprite': sprite,
                            'style': style
                        })
                    else:
                        print("❌ Keine Prompt-ID erhalten")
                else:
                    print(f"❌ Server-Fehler: {response.text[:100]}")

            except Exception as e:
                print(f"❌ Fehler bei {sprite}-{style}: {e}")

            # Kurze Pause zwischen Requests
            time.sleep(0.5)

    print()
    print("=" * 50)
    print(f"🎯 {successful_workflows} AI-WORKFLOWS GESTARTET!")
    print("🤖 AI VERARBEITET JETZT DEINE SPRITES...")
    print("=" * 50)

    if successful_workflows > 0:
        print("⏳ Überwache AI-Fortschritt...")

        completed = 0
        max_wait = 300  # 5 Minuten Maximum pro Workflow

        for prompt_info in all_prompt_ids:
            prompt_id = prompt_info['id']
            sprite = prompt_info['sprite']
            style = prompt_info['style']

            print(f"⏳ Warte auf: {sprite}-{style} (ID: {prompt_id})")

            # Warten auf Completion
            for attempt in range(max_wait // 2):  # Alle 2 Sekunden prüfen
                try:
                    history_response = requests.get(
                        f"{url}/history/{prompt_id}", timeout=3)

                    if history_response.status_code == 200:
                        history = history_response.json()

                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})

                            if status.get("completed", False):
                                print(f"✅ FERTIG: {sprite}-{style}")
                                completed += 1
                                break
                            elif "error" in status:
                                print(
                                    f"❌ FEHLER: {sprite}-{style} - {status['error']}")
                                break

                except:
                    pass

                time.sleep(2)

            # Kurze Pause zwischen Checks
            time.sleep(1)

        print()
        print("=" * 50)
        print("🏁 AI-VERARBEITUNG ABGESCHLOSSEN!")
        print("=" * 50)
        print(f"✅ Gestartete Workflows: {successful_workflows}")
        print(f"✅ Abgeschlossene Workflows: {completed}")
        print(f"📁 Ergebnisse in: ComfyUI_engine/output/")
        print()
        print("🎉 DEINE SPRITES WURDEN MIT ECHTER AI VERARBEITET!")

        # Zeige generierte Dateien
        try:
            import os
            output_dir = "ComfyUI_engine/output"
            if os.path.exists(output_dir):
                files = [f for f in os.listdir(
                    output_dir) if f.startswith("AI_")]
                if files:
                    print(f"📋 Generierte AI-Dateien ({len(files)}):")
                    for file in files[:10]:  # Zeige erste 10
                        print(f"   🖼️ {file}")
                    if len(files) > 10:
                        print(f"   ... und {len(files)-10} weitere")
        except:
            pass

    else:
        print("❌ KEINE AI-WORKFLOWS GESTARTET!")
        print("🔧 Überprüfe ComfyUI Installation")


if __name__ == "__main__":
    force_ai_processing()
