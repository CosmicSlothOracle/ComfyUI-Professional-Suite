#!/usr/bin/env python3
"""
SPRITE FRAME SPLITTER & BATCH PROCESSOR
Splits sprite sheets into individual frames for consistent SDXL processing
"""

import os
from PIL import Image
import json
import requests
from pathlib import Path


def split_idle_sprite():
    """Split the 9x1 idle sprite into 9 individual frames"""
    sprite_path = Path("input/sprite_sheets/idle_9_512x512_grid9x1.png")
    output_dir = Path("output/idle_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sprite_path.exists():
        print(f"ERROR: Sprite not found at {sprite_path}")
        return []

    try:
        # Load sprite sheet
        sprite_img = Image.open(sprite_path)
        width, height = sprite_img.size

        print(f"Original sprite: {width}x{height}")
        print(f"Expected: 9 frames in 9x1 grid")

        # Calculate frame size
        frame_width = width // 9  # 9 frames horizontal
        frame_height = height     # 1 frame vertical

        print(f"Each frame: {frame_width}x{frame_height}")

        frame_files = []

        # Split into individual frames
        for i in range(9):
            # Calculate crop box
            left = i * frame_width
            top = 0
            right = left + frame_width
            bottom = frame_height

            # Crop frame
            frame = sprite_img.crop((left, top, right, bottom))

            # Save frame
            frame_file = output_dir / f"idle_frame_{i+1:02d}.png"
            frame.save(frame_file)
            frame_files.append(frame_file)

            print(f"Frame {i+1}: {frame_file.name} ({frame.size})")

        print(f"\nSUCCESS: Split into {len(frame_files)} frames")
        print(f"Location: {output_dir}")

        return frame_files

    except Exception as e:
        print(f"ERROR: Failed to split sprite: {e}")
        return []


def upload_frames_to_comfyui(frame_files):
    """Upload all frames to ComfyUI"""
    uploaded_frames = []

    print(f"\nUploading {len(frame_files)} frames to ComfyUI...")

    for frame_file in frame_files:
        try:
            with open(frame_file, 'rb') as f:
                files = {'image': (frame_file.name, f, 'image/png')}
                response = requests.post(
                    "http://127.0.0.1:8188/upload/image", files=files)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name')
                uploaded_frames.append({
                    'original': frame_file.name,
                    'uploaded': uploaded_name
                })
                print(f"✅ {frame_file.name} → {uploaded_name}")
            else:
                print(f"❌ Failed to upload {frame_file.name}")

        except Exception as e:
            print(f"❌ Error uploading {frame_file.name}: {e}")

    print(f"\nUploaded {len(uploaded_frames)}/{len(frame_files)} frames")
    return uploaded_frames


def create_consistent_frame_workflow(uploaded_frame, frame_number, style):
    """Create workflow for single frame with LoRA consistency"""

    style_configs = {
        "anime": {
            "positive": "anime character, idle pose, cel shading, vibrant colors, 2d game art, clean lineart, detailed",
            "negative": "blurry, low quality, 3d, realistic, distorted, inconsistent style",
            "denoise": 0.4  # Medium denoise for good transformation
        },
        "pixel": {
            "positive": "pixel art character, idle pose, 16-bit style, retro game, crisp pixels, detailed sprite",
            "negative": "blurry, smooth, anti-aliased, high resolution, realistic",
            "denoise": 0.35
        },
        "enhanced": {
            "positive": "high quality character art, idle pose, detailed digital art, professional game sprite",
            "negative": "blurry, low quality, distorted, inconsistent",
            "denoise": 0.45
        }
    }

    config = style_configs.get(style, style_configs["anime"])

    # Single frame workflow with consistency
    workflow = {
        "1": {
            "inputs": {
                "image": uploaded_frame,
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
                "text": f"{config['positive']}, frame {frame_number} of 9, consistent character",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": config["negative"],
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
                "seed": 42 + frame_number,  # Consistent but varied seeds
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": config["denoise"],
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
                "filename_prefix": f"FRAME_{style}_{frame_number:02d}_",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow


def process_frames_with_consistency():
    """Process all frames with consistent style"""
    print("=" * 70)
    print("SPRITE FRAME SPLITTER & BATCH PROCESSOR")
    print("GOAL: Process individual frames for consistent results!")
    print("=" * 70)

    # Step 1: Split sprite
    print("\n1. SPLITTING SPRITE SHEET...")
    frame_files = split_idle_sprite()
    if not frame_files:
        return False

    # Step 2: Upload frames
    print("\n2. UPLOADING FRAMES...")
    uploaded_frames = upload_frames_to_comfyui(frame_files)
    if not uploaded_frames:
        return False

    # Step 3: Process each frame
    print("\n3. PROCESSING FRAMES WITH CONSISTENCY...")

    styles = ["anime", "pixel", "enhanced"]
    total_workflows = 0

    for style in styles:
        print(f"\n--- STYLE: {style.upper()} ---")

        for i, frame_data in enumerate(uploaded_frames):
            frame_number = i + 1
            uploaded_name = frame_data['uploaded']

            try:
                workflow = create_consistent_frame_workflow(
                    uploaded_name, frame_number, style)

                response = requests.post("http://127.0.0.1:8188/prompt",
                                         json={"prompt": workflow},
                                         timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    prompt_id = result.get("prompt_id")
                    print(f"✅ Frame {frame_number} ({style}): {prompt_id}")
                    total_workflows += 1
                else:
                    print(f"❌ Frame {frame_number} ({style}): Failed")

            except Exception as e:
                print(f"❌ Frame {frame_number} ({style}): {e}")

    print(f"\n" + "=" * 70)
    print("FRAME PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Started workflows: {total_workflows}")
    print(
        f"Expected results: {len(uploaded_frames)} frames × {len(styles)} styles = {len(uploaded_frames) * len(styles)} images")
    print("Results will be in: ComfyUI_engine/output/")
    print("Look for: FRAME_*.png")

    print("\n🎯 NEXT STEPS:")
    print("1. Wait for all frames to process")
    print("2. Collect FRAME_anime_*.png files")
    print("3. Reassemble into animated GIF")
    print("4. Compare with original sprite sheet")

    return total_workflows > 0


if __name__ == "__main__":
    process_frames_with_consistency()
