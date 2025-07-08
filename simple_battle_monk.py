#!/usr/bin/env python3
"""
SIMPLE BATTLE MONK PROCESSOR
Simplified workflow that definitely works
"""

import json
import os
import requests
from pathlib import Path


def upload_frames():
    """Upload all 12 kick attack frames to ComfyUI"""
    uploaded_names = []

    print("=== UPLOADING 12 KICK ATTACK FRAMES ===")

    for i in range(1, 13):
        frame_path = Path(f"ComfyUI_engine/input/frame ({i}).png")

        if not frame_path.exists():
            print(f"ERROR: Frame {i} not found at {frame_path}")
            continue

        try:
            with open(frame_path, 'rb') as f:
                files = {'image': (frame_path.name, f, 'image/png')}
                response = requests.post(
                    "http://127.0.0.1:8188/upload/image", files=files)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name', frame_path.name)
                uploaded_names.append(uploaded_name)
                print(f"SUCCESS: Frame {i}: {uploaded_name}")
            else:
                print(
                    f"ERROR: Frame {i}: Upload failed - Status {response.status_code}")
        except Exception as e:
            print(f"ERROR: Frame {i}: Upload error - {e}")

    print(f"\nUPLOADED: {len(uploaded_names)}/12 frames")
    return uploaded_names


def create_simple_workflow(frame_name, frame_number):
    """Create simple working workflow"""

    workflow = {
        "1": {
            "inputs": {
                "image": frame_name,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Frame {frame_number}"}
        },

        "2": {
            "inputs": {
                "ckpt_name": "dreamshaper_8.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Checkpoint"}
        },

        "3": {
            "inputs": {
                "text": f"pixel art, battle monk, fire kick attack, frame {frame_number} of 12, consistent character, flaming feet, martial arts, 16-bit style, pixel perfect, fighting game sprite, orange flames, red fire",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },

        "4": {
            "inputs": {
                "text": "blurry, realistic, 3d, photographic, smooth, anti-aliased, low quality",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },

        "5": {
            "inputs": {
                "seed": 42 + frame_number,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.4,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["6", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "Sampler"}
        },

        "6": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        },

        "7": {
            "inputs": {
                "samples": ["5", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },

        "8": {
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": f"MONK_FRAME_{frame_number:02d}",
                "format": "PNG",
                "quality": 100
            },
            "class_type": "SaveImage",
            "_meta": {"title": f"Save Frame {frame_number}"}
        }
    }

    return workflow


def execute_workflow(workflow, frame_number):
    """Execute workflow"""
    try:
        response = requests.post(
            "http://127.0.0.1:8188/prompt", json={"prompt": workflow})
        if response.status_code == 200:
            result = response.json()
            prompt_id = result.get('prompt_id')
            print(
                f"SUCCESS: Frame {frame_number} workflow started - ID: {prompt_id}")
            return prompt_id
        else:
            print(
                f"ERROR: Frame {frame_number} workflow failed - Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"ERROR: Frame {frame_number} workflow error: {e}")
        return None


def main():
    """Main function"""
    print("=== SIMPLE BATTLE MONK PROCESSOR ===")
    print("12-Frame Kick Attack -> Battle Monk with Fire Kicks")
    print()

    # Upload frames
    uploaded_names = upload_frames()
    if len(uploaded_names) == 0:
        print("FAILED: No frames uploaded")
        return

    print("\n=== CREATING WORKFLOWS ===")

    # Process each frame
    prompt_ids = []
    for i, frame_name in enumerate(uploaded_names, 1):
        print(f"\nProcessing Frame {i}...")

        workflow = create_simple_workflow(frame_name, i)
        prompt_id = execute_workflow(workflow, i)

        if prompt_id:
            prompt_ids.append(prompt_id)

    print(f"\n=== SUMMARY ===")
    print(
        f"Successfully started: {len(prompt_ids)}/{len(uploaded_names)} workflows")
    print("Expected results: MONK_FRAME_01_*.png through MONK_FRAME_12_*.png")

    if len(prompt_ids) > 0:
        print("\nBATTLE MONK TRANSFORMATION STARTED!")


if __name__ == "__main__":
    main()
