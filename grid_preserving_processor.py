#!/usr/bin/env python3
"""
GRID PRESERVING SPRITE PROCESSOR
Uses ControlNet + low denoise to PRESERVE sprite grid structure
"""

import json
import os
import requests
from pathlib import Path


def upload_sprite():
    """Upload Idle Sprite to ComfyUI"""
    sprite_path = Path("input/sprite_sheets/idle_9_512x512_grid9x1.png")

    if not sprite_path.exists():
        print("ERROR: Sprite not found at", sprite_path)
        return None

    try:
        with open(sprite_path, 'rb') as f:
            files = {'image': (sprite_path.name, f, 'image/png')}
            response = requests.post(
                "http://127.0.0.1:8188/upload/image", files=files)

        if response.status_code == 200:
            result = response.json()
            uploaded_name = result.get('name')
            print("SUCCESS: Sprite uploaded as", uploaded_name)
            return uploaded_name
        else:
            print("ERROR: Upload failed", response.text[:100])
            return None
    except Exception as e:
        print("ERROR: Upload exception", str(e))
        return None


def create_grid_preserving_workflow(uploaded_filename, style):
    """Create workflow that PRESERVES the grid structure"""

    if style == "anime":
        positive = "anime character, cel shading, vibrant colors, 2d game art, clean lineart"
        negative = "blurry, low quality, 3d, realistic, merged characters"
        denoise = 0.25  # MUCH LOWER!
    elif style == "pixel":
        positive = "pixel art character, 16-bit style, retro game, crisp pixels"
        negative = "blurry, smooth, anti-aliased, high resolution"
        denoise = 0.20  # EVEN LOWER!
    else:  # enhanced
        positive = "high quality character art, detailed, professional digital art"
        negative = "blurry, low quality, distorted"
        denoise = 0.30

    # GRID-PRESERVING WORKFLOW with ControlNet
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
                "text": positive,
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": negative,
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        # ControlNet für Struktur-Erhaltung
        "5": {
            "inputs": {
                "image": ["1", 0],
                "low_threshold": 100,
                "high_threshold": 200
            },
            "class_type": "Canny"
        },
        "6": {
            "inputs": {
                "control_net_name": "control_sd15_canny.pth"
            },
            "class_type": "ControlNetLoader"
        },
        "7": {
            "inputs": {
                "positive": ["3", 0],
                "control_net": ["6", 0],
                "image": ["5", 0],
                "strength": 0.8,  # Starke Struktur-Kontrolle
                "start_percent": 0.0,
                "end_percent": 1.0
            },
            "class_type": "ControlNetApply"
        },
        # VAE Encode mit Original
        "8": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode"
        },
        # KSampler mit NIEDRIGEM denoise
        "9": {
            "inputs": {
                "seed": 42 + hash(style) % 1000000,
                "steps": 20,  # Mehr Steps für bessere Qualität
                "cfg": 6.0,   # Niedrigere CFG
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise,  # KRITISCH: Niedrig!
                "model": ["2", 0],
                "positive": ["7", 0],  # Mit ControlNet
                "negative": ["4", 0],
                "latent_image": ["8", 0]
            },
            "class_type": "KSampler"
        },
        "10": {
            "inputs": {
                "samples": ["9", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEDecode"
        },
        "11": {
            "inputs": {
                "filename_prefix": f"GRID_PRESERVED_{style}_",
                "images": ["10", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def create_simple_low_denoise_workflow(uploaded_filename, style):
    """Fallback: Simple IMG2IMG with VERY low denoise"""

    if style == "anime":
        positive = "anime character sprite, cel shading, vibrant colors"
        negative = "blurry, low quality, 3d, realistic"
        denoise = 0.15  # SEHR NIEDRIG
    elif style == "pixel":
        positive = "pixel art character sprite, 16-bit style, crisp pixels"
        negative = "blurry, smooth, anti-aliased"
        denoise = 0.10  # EXTREM NIEDRIG
    else:  # enhanced
        positive = "high quality character sprite, detailed digital art"
        negative = "blurry, low quality, distorted"
        denoise = 0.20

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
                "text": positive,
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": negative,
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
                "seed": 12345 + hash(style) % 1000000,
                "steps": 15,
                "cfg": 5.0,  # Niedrige CFG
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": denoise,  # SEHR NIEDRIG!
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
                "filename_prefix": f"LOW_DENOISE_{style}_",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def process_with_structure_preservation():
    """Process sprite while preserving grid structure"""
    print("=" * 60)
    print("GRID PRESERVING SPRITE PROCESSOR")
    print("GOAL: Transform style while KEEPING grid structure!")
    print("=" * 60)

    # Upload sprite
    uploaded_filename = upload_sprite()
    if not uploaded_filename:
        return False

    styles = ["pixel", "anime", "enhanced"]
    successful = 0

    print("\nTesting STRUCTURE PRESERVATION approaches...")

    for style in styles:
        print(f"\n--- STYLE: {style.upper()} ---")

        # Try ControlNet approach first
        try:
            print("Trying ControlNet + Grid preservation...")
            workflow = create_grid_preserving_workflow(
                uploaded_filename, style)

            response = requests.post("http://127.0.0.1:8188/prompt",
                                     json={"prompt": workflow},
                                     timeout=10)

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(
                    f"SUCCESS: ControlNet workflow started (ID: {prompt_id})")
                successful += 1
            else:
                print(f"ControlNet failed: {response.text[:100]}")
                # Fallback to simple low denoise
                print("Falling back to low denoise approach...")
                workflow = create_simple_low_denoise_workflow(
                    uploaded_filename, style)

                response = requests.post("http://127.0.0.1:8188/prompt",
                                         json={"prompt": workflow},
                                         timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    print(
                        f"SUCCESS: Low denoise workflow started (ID: {prompt_id})")
                    successful += 1
                else:
                    print(f"Both approaches failed: {response.text[:100]}")

        except Exception as e:
            print(f"ERROR: Exception for {style}: {str(e)}")

    print(f"\n" + "=" * 60)
    print("GRID PRESERVATION PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Started workflows: {successful}/{len(styles)}")
    print("Results will be in: ComfyUI_engine/output/")
    print("Look for: GRID_PRESERVED_*.png and LOW_DENOISE_*.png")

    if successful > 0:
        print("\nTesting STRUCTURE-PRESERVING AI processing!")
        print("This should maintain your 9x1 grid layout!")

    return successful > 0


if __name__ == "__main__":
    process_with_structure_preservation()
