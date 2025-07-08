#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE IDLE SPRITE PROCESSOR
Verarbeitet: input/sprite_sheets/idle_9_512x512_grid9x1.png
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


def create_img2img_workflow(uploaded_filename, style):
    """Create IMG2IMG workflow for sprite transformation"""

    if style == "anime":
        positive = "anime character sprite sheet, idle animation, cel shading, vibrant colors, 2d game art"
        negative = "blurry, low quality, 3d, realistic, inconsistent frames"
        denoise = 0.6
    elif style == "pixel":
        positive = "pixel art sprite sheet, idle animation, 16-bit style, retro game, crisp pixels"
        negative = "blurry, smooth, anti-aliased, high resolution, realistic"
        denoise = 0.5
    else:  # enhanced
        positive = "high quality sprite sheet, idle animation, detailed game art, consistent frames"
        negative = "blurry, low quality, inconsistent, merged frames"
        denoise = 0.7

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
                "seed": 42 + hash(style) % 1000000,
                "steps": 12,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": denoise,
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
                "filename_prefix": f"IDLE_SPRITE_{style}_",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def process_idle_sprite():
    """Process the idle sprite with AI"""
    print("=" * 50)
    print("SIMPLE IDLE SPRITE PROCESSOR")
    print("TARGET: input/sprite_sheets/idle_9_512x512_grid9x1.png")
    print("=" * 50)

    # Check ComfyUI server
    try:
        response = requests.get(
            "http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code != 200:
            print("ERROR: ComfyUI server not available")
            return False
    except:
        print("ERROR: Cannot reach ComfyUI server")
        return False

    print("SUCCESS: ComfyUI server is ready")

    # Upload sprite
    print("\nUploading sprite to ComfyUI...")
    uploaded_filename = upload_sprite()
    if not uploaded_filename:
        return False

    # Process with different styles
    styles = ["anime", "pixel", "enhanced"]
    successful = 0

    print("\nStarting AI processing...")

    for style in styles:
        print(f"\nProcessing style: {style}")

        try:
            workflow = create_img2img_workflow(uploaded_filename, style)

            response = requests.post("http://127.0.0.1:8188/prompt",
                                     json={"prompt": workflow},
                                     timeout=10)

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(f"SUCCESS: AI workflow started (ID: {prompt_id})")
                successful += 1
            else:
                print(f"ERROR: Workflow failed: {response.text[:100]}")

        except Exception as e:
            print(f"ERROR: Exception for {style}: {str(e)}")

    print("\n" + "=" * 50)
    print("IDLE SPRITE PROCESSING COMPLETED")
    print("=" * 50)
    print(f"Started AI workflows: {successful}/{len(styles)}")
    print("Results will be in: ComfyUI_engine/output/")
    print("Look for: IDLE_SPRITE_*.png")

    if successful > 0:
        print("\nAI is now processing your idle sprite!")
        print("Wait a few minutes for results...")
        print("Your original 9x1 grid will be transformed!")

    return successful > 0


if __name__ == "__main__":
    process_idle_sprite()
