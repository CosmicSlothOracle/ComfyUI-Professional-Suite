#!/usr/bin/env python3
"""
VAPORWAVE WORKFLOW 3: GLITCH CYBERPUNK
Transform transparent videos into glitchy cyberpunk vaporwave style
- Digital glitch effects
- Datamoshing
- Chromatic aberration
- Cyberpunk color schemes
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path


def create_vaporwave_glitch_workflow():
    """Create ComfyUI workflow for glitch cyberpunk vaporwave style"""

    workflow = {
        "nodes": {
            "1": {
                "inputs": {"video": "input_video_path"},
                "class_type": "VHS_VideoLoader"
            },
            "2": {
                "inputs": {
                    "image": ["1", 0],
                    "glitch_type": "digital_corruption",
                    "intensity": 0.6,
                    "block_size": 8,
                    "frequency": 0.3
                },
                "class_type": "ImageGlitchEffect"
            },
            "3": {
                "inputs": {
                    "image": ["2", 0],
                    "aberration_strength": 8,
                    "red_offset": [3, 0],
                    "green_offset": [0, 0],
                    "blue_offset": [-3, 0]
                },
                "class_type": "ImageChromaticAberration"
            },
            "4": {
                "inputs": {
                    "image": ["3", 0],
                    "hue_shift": 0.7,  # Towards cyan/magenta
                    "saturation": 2.0,
                    "contrast": 1.5,
                    "brightness": 0.9
                },
                "class_type": "ImageColorAdjust"
            },
            "5": {
                "inputs": {
                    "image": ["4", 0],
                    "datamosh_strength": 0.4,
                    "compression_artifacts": True,
                    "pixel_sorting": 0.3
                },
                "class_type": "ImageDatamoshEffect"
            },
            "6": {
                "inputs": {
                    "image": ["5", 0],
                    "scanline_intensity": 0.5,
                    "scanline_spacing": 2,
                    "rgb_shift": True,
                    "static_noise": 0.2
                },
                "class_type": "ImageCRTEffect"
            },
            "7": {
                "inputs": {
                    "image": ["6", 0],
                    "overlay_type": "cyberpunk_grid",
                    "grid_color": [0, 255, 255],  # Cyan
                    "opacity": 0.4,
                    "animation": "pulse"
                },
                "class_type": "ImageCyberpunkOverlay"
            },
            "8": {
                "inputs": {
                    "image": ["7", 0],
                    "text": "ERROR 404",
                    "font": "cyberpunk_mono",
                    "color": [255, 0, 255],  # Magenta
                    "glitch_text": True,
                    "position": "random"
                },
                "class_type": "ImageGlitchText"
            },
            "9": {
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": 30,
                    "format": "mp4",
                    "compression": "high_quality",
                    "filename_prefix": "vaporwave_glitch_"
                },
                "class_type": "VHS_VideoOutput"
            }
        }
    }

    return workflow


def process_vaporwave_glitch(input_file, output_dir):
    """Process a single file with glitch vaporwave style"""

    print(f"⚡ GLITCH VAPORWAVE: {input_file.name}")

    # ComfyUI processing command
    cmd = [
        sys.executable, "main.py",
        "--workflow", "vaporwave_glitch",
        "--input", str(input_file),
        "--output-dir", str(output_dir),
        "--extra-args", json.dumps({
            "glitch_intensity": 0.6,
            "chromatic_aberration": 8,
            "datamosh_strength": 0.4,
            "cyberpunk_mode": True
        })
    ]

    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Vaporwave Glitch Style Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--glitch-intensity", type=float,
                        default=0.6, help="Glitch effect intensity")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    result = process_vaporwave_glitch(input_file, output_dir)

    if result.returncode == 0:
        print("✅ GLITCH VAPORWAVE SUCCESS!")
    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
