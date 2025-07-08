#!/usr/bin/env python3
"""
BATTLE MONK PIXEL ART PROCESSOR
Transforms 12-frame kick attack → Pixel Art Battle Monk with fire kicks
Uses optimal workflow configuration for maximum consistency
"""

import json
import os
import requests
from pathlib import Path


def upload_frames():
    """Upload all 12 kick attack frames to ComfyUI"""
    frame_paths = []
    uploaded_names = []

    # Define frame paths
    for i in range(1, 13):
        frame_path = Path(f"ComfyUI_engine/input/frame ({i}).png")
        frame_paths.append(frame_path)

    print("=== UPLOADING 12 KICK ATTACK FRAMES ===")

    for i, frame_path in enumerate(frame_paths, 1):
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
                print(f"✅ Frame {i}: {uploaded_name}")
            else:
                print(
                    f"❌ Frame {i}: Upload failed - Status {response.status_code}")
        except Exception as e:
            print(f"❌ Frame {i}: Error uploading - {e}")

    print(f"\n✅ UPLOADED: {len(uploaded_names)}/12 frames")
    return uploaded_names


def create_battle_monk_workflow(frame_name, frame_number):
    """Create optimal battle monk workflow for single frame"""

    # OPTIMAL CONFIGURATION based on analysis
    workflow = {
        "1": {
            "inputs": {
                "image": frame_name,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Kick Frame {frame_number}"}
        },

        # OPTIMAL CHECKPOINT for pixel art
        "2": {
            "inputs": {
                "ckpt_name": "realisticVisionV60B1_v51VAE.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Optimal Pixel Art Checkpoint"}
        },

        # CONTROLNET for structure preservation
        "3": {
            "inputs": {
                "control_net_name": "control_v11p_sd15_canny.pth"
            },
            "class_type": "ControlNetLoader",
            "_meta": {"title": "ControlNet Structure Preservation"}
        },

        # CANNY EDGE DETECTION
        "4": {
            "inputs": {
                "image": ["1", 0],
                "low_threshold": 100,
                "high_threshold": 200
            },
            "class_type": "CannyEdgePreprocessor",
            "_meta": {"title": "Canny Edge Detection"}
        },

        # BATTLE MONK POSITIVE CONDITIONING
        "5": {
            "inputs": {
                "text": f"pixel art, 16-bit style, battle monk character, fire kick attack, frame {frame_number} of 12, consistent character design, flaming feet, martial arts pose, pixel perfect, sharp edges, limited color palette, fire effects, orange flames, red fire, action pose, fighting game sprite",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Battle Monk Positive Prompt"}
        },

        # NEGATIVE CONDITIONING
        "6": {
            "inputs": {
                "text": "blurry, realistic, 3d, photographic, soft edges, anti-aliased, low quality, distorted, noise, smooth, gradient",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },

        # CONTROLNET APPLY
        "7": {
            "inputs": {
                "conditioning": ["5", 0],
                "control_net": ["3", 0],
                "image": ["4", 0],
                "strength": 0.8  # Strong structure preservation
            },
            "class_type": "ControlNetApply",
            "_meta": {"title": "Apply ControlNet"}
        },

        # VAE ENCODE
        "8": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        },

        # OPTIMAL SAMPLER CONFIGURATION
        "9": {
            "inputs": {
                "seed": 42 + frame_number,  # Sequential seeds for consistency
                "steps": 25,  # Optimal steps
                "cfg": 7.5,   # Optimal CFG
                "sampler_name": "dpmpp_2m_sde",  # Optimal sampler
                "scheduler": "karras",  # Optimal scheduler
                "denoise": 0.35,  # Optimal denoise for structure preservation
                "model": ["2", 0],
                "positive": ["7", 0],  # ControlNet conditioned
                "negative": ["6", 0],
                "latent_image": ["8", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "Optimal Battle Monk Sampler"}
        },

        # VAE DECODE
        "10": {
            "inputs": {
                "samples": ["9", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },

        # SAVE BATTLE MONK FRAME
        "11": {
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": f"BATTLE_MONK_FRAME_{frame_number:02d}",
                "format": "PNG",
                "quality": 100
            },
            "class_type": "SaveImage",
            "_meta": {"title": f"Save Battle Monk Frame {frame_number}"}
        }
    }

    return workflow


def execute_workflow(workflow, frame_number):
    """Execute workflow for single frame"""
    try:
        response = requests.post(
            "http://127.0.0.1:8188/prompt", json={"prompt": workflow})
        if response.status_code == 200:
            result = response.json()
            prompt_id = result.get('prompt_id')
            print(
                f"✅ Frame {frame_number}: Workflow started - ID: {prompt_id}")
            return prompt_id
        else:
            print(
                f"❌ Frame {frame_number}: Workflow failed - Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Frame {frame_number}: Error executing workflow - {e}")
        return None


def main():
    """Main execution function"""
    print("=== BATTLE MONK PIXEL ART PROCESSOR ===")
    print("12-Frame Kick Attack → Pixel Art Battle Monk with Fire Kicks")
    print("Using OPTIMAL workflow configuration for maximum consistency")
    print()

    # Upload all frames
    uploaded_names = upload_frames()
    if len(uploaded_names) != 12:
        print(
            f"WARNING: Only {len(uploaded_names)}/12 frames uploaded successfully")
        if len(uploaded_names) == 0:
            print("FAILED: No frames uploaded")
            return

    print("\n=== CREATING BATTLE MONK WORKFLOWS ===")

    # Create and execute workflows for each frame
    prompt_ids = []
    for i, frame_name in enumerate(uploaded_names, 1):
        print(f"\nProcessing Frame {i}/12...")

        # Create workflow for this frame
        workflow = create_battle_monk_workflow(frame_name, i)

        # Execute workflow
        prompt_id = execute_workflow(workflow, i)
        if prompt_id:
            prompt_ids.append(prompt_id)

    print("\n=== EXECUTION SUMMARY ===")
    print(f"✅ Successfully started: {len(prompt_ids)}/12 workflows")
    print("\nOPTIMAL CONFIGURATION USED:")
    print("- Checkpoint: realisticVisionV60B1_v51VAE.safetensors")
    print("- ControlNet: control_v11p_sd15_canny.pth (strength 0.8)")
    print("- Sampler: DPM++ 2M SDE Karras")
    print("- Steps: 25, CFG: 7.5, Denoise: 0.35")
    print("- Sequential seeds: 43-54 for consistency")
    print("\nExpected results:")
    print("- BATTLE_MONK_FRAME_01_*.png through BATTLE_MONK_FRAME_12_*.png")
    print("- Pixel art battle monk with fire kick effects")
    print("- Consistent character design across all frames")
    print("- Structure preserved from original kick animation")

    if len(prompt_ids) > 0:
        print(f"\n🔥 BATTLE MONK TRANSFORMATION IN PROGRESS! 🔥")
        print("AI is creating your fire-kicking pixel art battle monk...")


if __name__ == "__main__":
    main()
