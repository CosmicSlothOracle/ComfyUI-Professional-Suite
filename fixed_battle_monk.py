#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXED BATTLE MONK PROCESSOR
Uses ONLY proven working configuration - no experimental nodes
"""

import json
import os
import requests
from pathlib import Path


def upload_frame(frame_number):
    """Upload single frame to ComfyUI"""
    frame_path = Path(f"ComfyUI_engine/input/frame ({frame_number}).png")

    if not frame_path.exists():
        print(f"ERROR: Frame {frame_number} not found")
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
            print(f"ERROR: Frame {frame_number} upload failed")
            return None
    except Exception as e:
        print(f"ERROR: Frame {frame_number} exception: {str(e)}")
        return None


def create_working_workflow(uploaded_filename, frame_number):
    """Create EXACTLY the working workflow structure"""

    positive = f"pixel art, battle monk character, fire kick attack, frame {frame_number} of 12, consistent character design, flaming feet, martial arts pose, 16-bit style, fighting game sprite, orange flames, red fire"
    negative = "blurry, realistic, 3d, photographic, soft edges, anti-aliased, low quality, distorted, smooth"

    # EXACT WORKING STRUCTURE - NO MODIFICATIONS
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
                "ckpt_name": "sdxl.safetensors"  # ONLY checkpoint that works
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
                "seed": 42 + frame_number,
                "steps": 12,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.5,
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
                "filename_prefix": f"BATTLE_MONK_F{frame_number:02d}_",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def process_battle_monk():
    """Process battle monk frames with WORKING configuration"""
    print("=== FIXED BATTLE MONK PROCESSOR ===")
    print("Using ONLY proven working configuration")
    print()

    # Check server
    try:
        response = requests.get(
            "http://127.0.0.1:8188/system_stats", timeout=5)
        if response.status_code != 200:
            print("ERROR: ComfyUI server not ready")
            return False
    except:
        print("ERROR: Cannot reach ComfyUI server")
        return False

    print("SUCCESS: ComfyUI server ready")

    # Process frames 1-12
    successful = 0

    for frame_num in range(1, 13):
        print(f"\n--- Processing Frame {frame_num}/12 ---")

        # Upload frame
        uploaded_filename = upload_frame(frame_num)
        if not uploaded_filename:
            continue

        # Create and execute workflow
        try:
            workflow = create_working_workflow(uploaded_filename, frame_num)

            response = requests.post("http://127.0.0.1:8188/prompt",
                                     json={"prompt": workflow},
                                     timeout=10)

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                print(f"SUCCESS: Frame {frame_num} started (ID: {prompt_id})")
                successful += 1
            else:
                print(f"ERROR: Frame {frame_num} failed: {response.text[:50]}")

        except Exception as e:
            print(f"ERROR: Frame {frame_num} exception: {str(e)}")

    print(f"\n=== RESULTS ===")
    print(f"Successfully started: {successful}/12 workflows")
    print("Look for: BATTLE_MONK_F01_*.png to BATTLE_MONK_F12_*.png")

    if successful > 0:
        print("\nBATTLE MONK TRANSFORMATION STARTED!")
        print("Using ONLY working configuration - no experimental nodes!")

    return successful > 0


if __name__ == "__main__":
    process_battle_monk()
