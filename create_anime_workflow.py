#!/usr/bin/env python3
"""
ANIME WORKFLOW GENERATOR - ULTIMATE PIPELINE
===========================================

Erstellt sofort einsatzbereite ComfyUI Workflows für Anime Video Generation.
Alle Modelle sind heruntergeladen und bereit!
"""

import json
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnimeWorkflowGenerator:
    """
    Generiert einsatzbereite ComfyUI Workflows für Anime Videos
    """

    def __init__(self):
        self.output_dir = Path("workflows")
        self.output_dir.mkdir(exist_ok=True)

        # Basis-Workflow für Anime Video Generation
        self.base_workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": "counterfeitV30_v30.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "2": {
                "inputs": {
                    "text": "masterpiece, best quality, 1girl, anime style, beautiful face, detailed eyes, long hair, school uniform, cherry blossoms, spring, soft lighting, cinematic",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "3": {
                "inputs": {
                    "text": "worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, multiple eyebrow, cropped, extra limb, missing limbs, deformed hands, long neck, long body, bad hands, signature, username, artist name, conjoined fingers, deformed fingers, ugly eyes, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, painting by bad-artist",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Negative)"
                }
            },
            "4": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 16
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "5": {
                "inputs": {
                    "model_name": "v3_sd15_mm.ckpt"
                },
                "class_type": "ADE_AnimateDiffLoaderGen1",
                "_meta": {
                    "title": "AnimateDiff Loader"
                }
            },
            "6": {
                "inputs": {
                    "model": ["1", 0],
                    "motion_model": ["5", 0],
                    "beta_schedule": "sqrt_linear (AnimateDiff)"
                },
                "class_type": "ADE_UseEvolvedSampling",
                "_meta": {
                    "title": "Use Evolved Sampling"
                }
            },
            "7": {
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "8": {
                "inputs": {
                    "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "9": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["8", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "10": {
                "inputs": {
                    "frame_rate": 8,
                    "loop_count": 0,
                    "filename_prefix": "anime_video",
                    "format": "image/gif",
                    "pix_fmt": "rgb24",
                    "crf": 20,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["9", 0]
                },
                "class_type": "VHS_VideoCombine",
                "_meta": {
                    "title": "Video Combine"
                }
            }
        }

    def create_basic_anime_workflow(self):
        """
        Erstellt einen grundlegenden Anime-Workflow
        """
        workflow_path = self.output_dir / "basic_anime_video.json"

        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(self.base_workflow, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Basic Anime Workflow erstellt: {workflow_path}")
        return workflow_path

    def create_character_focused_workflow(self):
        """
        Erstellt einen Charakter-fokussierten Workflow
        """
        character_workflow = self.base_workflow.copy()

        # Angepasster Prompt für Charaktere
        character_workflow["2"]["inputs"]["text"] = "masterpiece, best quality, 1girl, anime style, beautiful detailed face, expressive eyes, dynamic hair movement, school uniform, confident pose, detailed background, professional lighting, high resolution"

        # Höhere Qualitätseinstellungen
        character_workflow["7"]["inputs"]["steps"] = 25
        character_workflow["7"]["inputs"]["cfg"] = 8.0

        workflow_path = self.output_dir / "character_focused_anime.json"

        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(character_workflow, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Character Focused Workflow erstellt: {workflow_path}")
        return workflow_path

    def create_action_scene_workflow(self):
        """
        Erstellt einen Action-Szenen Workflow
        """
        action_workflow = self.base_workflow.copy()

        # Action-fokussierter Prompt
        action_workflow["2"]["inputs"]["text"] = "masterpiece, best quality, dynamic action scene, anime style, fighting pose, motion blur, speed lines, dramatic lighting, intense expression, detailed animation, high energy"

        # Mehr Frames für flüssige Action
        action_workflow["4"]["inputs"]["batch_size"] = 24
        action_workflow["10"]["inputs"]["frame_rate"] = 12

        workflow_path = self.output_dir / "action_scene_anime.json"

        with open(workflow_path, 'w', encoding='utf-8') as f:
            json.dump(action_workflow, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Action Scene Workflow erstellt: {workflow_path}")
        return workflow_path

    def create_workflow_guide(self):
        """
        Erstellt eine Anleitung für die Workflows
        """
        guide_content = """
# ANIME WORKFLOW GUIDE - ULTIMATE PIPELINE
========================================

## VERFÜGBARE WORKFLOWS:

### 1. BASIC ANIME VIDEO (basic_anime_video.json)
- Einfacher Einstieg in Anime Video Generation
- 16 Frames, 8 FPS
- Standard-Qualitätseinstellungen
- Perfekt für erste Tests

### 2. CHARACTER FOCUSED (character_focused_anime.json)
- Optimiert für Charakter-Animationen
- Höhere Qualitätseinstellungen (25 Steps, CFG 8.0)
- Fokus auf Gesichtsdetails und Ausdrücke
- Ideal für Charakter-Präsentationen

### 3. ACTION SCENE (action_scene_anime.json)
- Optimiert für Action-Szenen
- 24 Frames, 12 FPS für flüssige Bewegungen
- Motion Blur und Speed Lines
- Perfekt für dynamische Szenen

## VERWENDUNG:

1. **ComfyUI öffnen**
2. **Workflow laden**: File → Load → [Workflow-Datei auswählen]
3. **Prompt anpassen**: Text in Node 2 (Positive Prompt) ändern
4. **Parameter anpassen** (optional):
   - Seed für verschiedene Variationen
   - Steps für Qualität (mehr = besser, aber langsamer)
   - CFG für Prompt-Adherence
5. **Queue Prompt** klicken
6. **Warten** (ca. 2-5 Minuten je nach Hardware)
7. **Ergebnis** im Output-Ordner finden

## PROMPT-TIPPS:

### QUALITÄTS-TAGS (IMMER VERWENDEN):
- masterpiece, best quality
- high resolution, detailed
- anime style, professional

### CHARAKTER-BESCHREIBUNG:
- 1girl/1boy, beautiful face
- detailed eyes, expressive
- hair color/style, clothing

### SZENEN-BESCHREIBUNG:
- background, lighting
- mood, atmosphere
- camera angle, composition

### NEGATIVE PROMPTS (VERMEIDEN):
- worst quality, low quality
- bad anatomy, deformed
- blurry, pixelated

## TROUBLESHOOTING:

### FEHLER: "Model not found"
- Überprüfe, ob alle Modelle heruntergeladen sind
- Starte ComfyUI neu

### LANGSAME GENERATION:
- Reduziere Batch Size (weniger Frames)
- Reduziere Steps (aber nicht unter 15)
- Reduziere Resolution (512x512 → 256x256)

### SCHLECHTE QUALITÄT:
- Erhöhe Steps (20 → 30)
- Verbessere Prompts
- Experimentiere mit verschiedenen Seeds

## ERWEITERTE EINSTELLUNGEN:

### FRAME-ANZAHL:
- 8 Frames: Kurze Loops
- 16 Frames: Standard-Animationen
- 24+ Frames: Längere Sequenzen

### FRAME RATE:
- 6 FPS: Langsame, ruhige Bewegungen
- 8 FPS: Standard-Animationen
- 12+ FPS: Flüssige, schnelle Bewegungen

### AUFLÖSUNG:
- 256x256: Schnelle Tests
- 512x512: Standard-Qualität
- 768x768: Hohe Qualität (erfordert mehr VRAM)

## NÄCHSTE SCHRITTE:
1. Teste alle drei Workflows
2. Experimentiere mit verschiedenen Prompts
3. Passe Parameter an deine Hardware an
4. Erstelle eigene Variationen

Viel Spaß beim Anime-Video-Erstellen! 🎌✨
"""

        guide_path = self.output_dir / "WORKFLOW_GUIDE.md"

        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        logger.info(f"✅ Workflow Guide erstellt: {guide_path}")
        return guide_path

    def generate_all_workflows(self):
        """
        Generiert alle Workflows und die Anleitung
        """
        logger.info("🚀 GENERIERE ALLE ANIME WORKFLOWS...")

        workflows = []
        workflows.append(self.create_basic_anime_workflow())
        workflows.append(self.create_character_focused_workflow())
        workflows.append(self.create_action_scene_workflow())
        guide = self.create_workflow_guide()

        logger.info(f"\n🎯 ALLE WORKFLOWS ERFOLGREICH ERSTELLT!")
        logger.info(f"📁 Speicherort: {self.output_dir.absolute()}")
        logger.info(f"📋 Anleitung: {guide}")

        return workflows, guide


def main():
    """
    Hauptfunktion
    """
    print("🎌 ANIME WORKFLOW GENERATOR - ULTIMATE PIPELINE")
    print("=" * 60)
    print("Erstellt sofort einsatzbereite ComfyUI Workflows")
    print("Alle Modelle sind heruntergeladen und bereit!")
    print("=" * 60)

    try:
        generator = AnimeWorkflowGenerator()
        workflows, guide = generator.generate_all_workflows()

        print(f"\n🎯 WORKFLOWS ERFOLGREICH ERSTELLT!")
        print(f"\n📁 VERFÜGBARE WORKFLOWS:")
        for workflow in workflows:
            print(f"   ✅ {workflow.name}")

        print(f"\n📋 ANLEITUNG: {guide.name}")

        print(f"\n🚀 NÄCHSTE SCHRITTE:")
        print("   1. ComfyUI öffnen")
        print("   2. Workflow laden (File → Load)")
        print("   3. Prompt anpassen")
        print("   4. Queue Prompt klicken")
        print("   5. Erste Anime-Videos genießen! 🎌✨")

        return True

    except Exception as e:
        logger.error(f"❌ Fehler beim Erstellen der Workflows: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
