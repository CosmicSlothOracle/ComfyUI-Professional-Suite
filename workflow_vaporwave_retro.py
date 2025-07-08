#!/usr/bin/env python3
"""
VAPORWAVE WORKFLOW 2: RETRO SUNSET
Transform transparent videos into retro sunset vaporwave style
- Sunset color palettes
- Palm tree silhouettes
- VHS degradation effects
- Retro typography overlays
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path


def create_vaporwave_retro_workflow():
    """Create ComfyUI workflow for retro sunset vaporwave style"""

    workflow = {
        "nodes": {
            "1": {
                "inputs": {"video": "input_video_path"},
                "class_type": "VHS_VideoLoader"
            },
            "2": {
                "inputs": {
                    "image": ["1", 0],
                    "gradient_type": "sunset",
                    "colors": ["#ff6b35", "#f7931e", "#ffd23f", "#ff006e", "#8338ec"],
                    "direction": "vertical",
                    "opacity": 0.6
                },
                "class_type": "ImageGradientOverlay"
            },
            "3": {
                "inputs": {
                    "image": ["2", 0],
                    "silhouette_type": "palm_trees",
                    "position": "bottom",
                    "scale": 0.8,
                    "opacity": 0.9
                },
                "class_type": "ImageSilhouetteOverlay"
            },
            "4": {
                "inputs": {
                    "image": ["3", 0],
                    "vhs_noise": 0.4,
                    "tracking_errors": True,
                    "color_bleeding": 0.3,
                    "tape_wear": 0.5
                },
                "class_type": "ImageVHSEffect"
            },
            "5": {
                "inputs": {
                    "image": ["4", 0],
                    "font": "retro_computer",
                    "text": "VAPOR WAVE",
                    "color": [255, 255, 255],
                    "position": "top_center",
                    "glow": True
                },
                "class_type": "ImageTextOverlay"
            },
            "6": {
                "inputs": {
                    "image": ["5", 0],
                    "sun_position": "center_top",
                    "sun_size": 0.3,
                    "rays": True,
                    "color": "#ffd23f"
                },
                "class_type": "ImageSunOverlay"
            },
            "7": {
                "inputs": {
                    "image": ["6", 0],
                    "contrast": 1.3,
                    "warmth": 0.4,
                    "vintage_tone": True
                },
                "class_type": "ImageColorGrading"
            },
            "8": {
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": 24,  # Retro feel
                    "format": "mp4",
                    "filename_prefix": "vaporwave_retro_"
                },
                "class_type": "VHS_VideoOutput"
            }
        }
    }

    return workflow


def process_vaporwave_retro(input_file, output_dir):
    """Process a single file with retro vaporwave style"""

    print(f"🌅 RETRO VAPORWAVE: {input_file.name}")

    # ComfyUI processing command
    cmd = [
        sys.executable, "main.py",
        "--workflow", "vaporwave_retro",
        "--input", str(input_file),
        "--output-dir", str(output_dir),
        "--extra-args", json.dumps({
            "sunset_intensity": 0.6,
            "vhs_degradation": 0.4,
            "palm_trees": True,
            "retro_text": "VAPOR WAVE"
        })
    ]

    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Vaporwave Retro Style Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--text", default="VAPOR WAVE", help="Overlay text")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    result = process_vaporwave_retro(input_file, output_dir)

    if result.returncode == 0:
        print("✅ RETRO VAPORWAVE SUCCESS!")
    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
