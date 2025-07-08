#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WORKING BATTLE MONK PROCESSOR
Uses proven successful workflow structure
"""

import json
import os
import requests
from pathlib import Path


def upload_frame(frame_number):
    """Upload single frame to ComfyUI"""
    frame_path = Path(f"ComfyUI_engine/input/frame ({frame_number}).png")

    if not frame_path.exists():
        print(f"ERROR: Frame {frame_number} not found at {frame_path}")
        return None

    try:
        with open(frame_path, 'rb') as f:
            files = {'image': (frame_path.name, f, 'image/png')}
            response = requests.post(
                "http://127.0.0.1:8188/upload/image", files=files)

        if response.status_code == 200:
            result = response.json()
            uploaded_name = result.get('name')
            print(f"SUCCESS: Frame {frame_number} uploaded as {uploaded_name}")
            return uploaded_name
        else:
            print(
                f"ERROR: Frame {frame_number} upload failed {response.text[:100]}")
            return None
    except Exception as e:
        print(f"ERROR: Frame {frame_number} upload exception {str(e)}")
        return None


def create_battle_monk_workflow(uploaded_filename, frame_number):
    """Create battle monk workflow using proven structure"""

    positive = f"pixel art, battle monk character, fire kick attack, frame {frame_number} of 12, consistent character design, flaming feet, martial arts pose, 16-bit style, fighting game sprite, orange flames, red fire, action pose"
    negative = "blurry, realistic, 3d, photographic, soft edges, anti-aliased, low quality, distorted, smooth"

    # EXACT WORKING STRUCTURE from simple_idle_processor.py
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
                "seed": 42 + frame_number,  # Sequential seeds for consistency
                "steps": 12,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.5,  # Good for pixel art
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
                "filename_prefix": f"BATTLE_MONK_FRAME_{frame_number:02d}_",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def process_battle_monk_frames():
    """Process all 12 battle monk frames"""
    print("=" * 60)
    print("WORKING BATTLE MONK PROCESSOR")
    print("12-Frame Kick Attack -> Battle Monk with Fire Kicks")
    print("=" * 60)

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

    # Process each frame
    successful = 0
    total_frames = 12

    print(f"\nProcessing {total_frames} kick attack frames...")

    for frame_num in range(1, total_frames + 1):
        print(f"\n--- FRAME {frame_num}/{total_frames} ---")

        # Upload frame
        uploaded_filename = upload_frame(frame_num)
        if not uploaded_filename:
            continue

        # Create and execute workflow
        try:
            workflow = create_battle_monk_workflow(
                uploaded_filename, frame_num)

            response = requests.post("http://127.0.0.1:8188/prompt",
                                     json={"prompt": workflow},
                                     timeout=10)

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(
                    f"SUCCESS: Frame {frame_num} workflow started (ID: {prompt_id})")
                successful += 1
            else:
                print(
                    f"ERROR: Frame {frame_num} workflow failed: {response.text[:100]}")

        except Exception as e:
            print(f"ERROR: Frame {frame_num} exception: {str(e)}")

    print("\n" + "=" * 60)
    print("BATTLE MONK PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Started AI workflows: {successful}/{total_frames}")
    print("Results will be in: ComfyUI_engine/output/")
    print("Look for: BATTLE_MONK_FRAME_01_*.png through BATTLE_MONK_FRAME_12_*.png")

    if successful > 0:
        print("\nBATTLE MONK TRANSFORMATION IN PROGRESS!")
        print("AI is converting your kick attack into fire-kicking battle monk!")
        print("Wait a few minutes for results...")
        print("Each frame will be individually transformed for consistency!")

    return successful > 0


if __name__ == "__main__":
    process_battle_monk_frames()
