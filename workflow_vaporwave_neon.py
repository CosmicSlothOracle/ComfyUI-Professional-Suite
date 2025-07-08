#!/usr/bin/env python3
"""
VAPORWAVE WORKFLOW 1: NEON AESTHETICS
Transform transparent videos into vibrant neon vaporwave style
- Neon pink/cyan color schemes
- Glowing effects
- Grid overlays
- 80s retro vibes
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path


def create_vaporwave_neon_workflow():
    """Create ComfyUI workflow for neon vaporwave style"""

    workflow = {
        "nodes": {
            "1": {
                "inputs": {"video": "input_video_path"},
                "class_type": "VHS_VideoLoader"
            },
            "2": {
                "inputs": {
                    "image": ["1", 0],
                    "hue": 0.3,  # Shift towards pink/magenta
                    "saturation": 1.8,  # High saturation
                    "value": 1.2  # Bright
                },
                "class_type": "ImageColorAdjust"
            },
            "3": {
                "inputs": {
                    "image": ["2", 0],
                    "filter_type": "neon_glow",
                    "intensity": 0.8,
                    "glow_radius": 15
                },
                "class_type": "ImageFilter"
            },
            "4": {
                "inputs": {
                    "image": ["3", 0],
                    "overlay_type": "grid",
                    "grid_size": 32,
                    "grid_color": [255, 0, 255],  # Magenta
                    "opacity": 0.3
                },
                "class_type": "ImageOverlay"
            },
            "5": {
                "inputs": {
                    "image": ["4", 0],
                    "style": "synthwave_gradient",
                    # Pink to purple to cyan
                    "colors": ["#ff006e", "#8338ec", "#3a86ff"],
                    "direction": "diagonal"
                },
                "class_type": "ImageGradientOverlay"
            },
            "6": {
                "inputs": {
                    "image": ["5", 0],
                    "chromatic_aberration": 0.4,
                    "vhs_noise": 0.2,
                    "scanlines": True
                },
                "class_type": "ImageVintageEffect"
            },
            "7": {
                "inputs": {
                    "images": ["6", 0],
                    "frame_rate": 30,
                    "format": "mp4",
                    "filename_prefix": "vaporwave_neon_"
                },
                "class_type": "VHS_VideoOutput"
            }
        }
    }

    return workflow


def process_vaporwave_neon(input_file, output_dir):
    """Process a single file with neon vaporwave style"""

    print(f"🌈 NEON VAPORWAVE: {input_file.name}")

    # ComfyUI processing command
    cmd = [
        sys.executable, "main.py",
        "--workflow", "vaporwave_neon",
        "--input", str(input_file),
        "--output-dir", str(output_dir),
        "--extra-args", json.dumps({
            "neon_intensity": 0.8,
            "grid_opacity": 0.3,
            "color_shift": "pink_cyan",
            "glow_radius": 15
        })
    ]

    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Vaporwave Neon Style Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    result = process_vaporwave_neon(input_file, output_dir)

    if result.returncode == 0:
        print("✅ NEON VAPORWAVE SUCCESS!")
    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
