#!/usr/bin/env python3
"""
VAPORWAVE WORKFLOW 4: CHROME GEOMETRY
Transform transparent videos into chrome metallic vaporwave style
- Metallic reflective surfaces
- Geometric overlays
- Chrome gradients
- 3D geometric shapes
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path


def create_vaporwave_chrome_workflow():
    """Create ComfyUI workflow for chrome geometric vaporwave style"""

    workflow = {
        "nodes": {
            "1": {
                "inputs": {"video": "input_video_path"},
                "class_type": "VHS_VideoLoader"
            },
            "2": {
                "inputs": {
                    "image": ["1", 0],
                    "metallic_effect": "chrome",
                    "reflection_intensity": 0.8,
                    "shininess": 0.9,
                    "environment_map": "abstract"
                },
                "class_type": "ImageMetallicEffect"
            },
            "3": {
                "inputs": {
                    "image": ["2", 0],
                    "gradient_type": "chrome_spectrum",
                    "colors": ["#c0c0c0", "#ffffff", "#e6e6fa", "#ff69b4", "#00ffff"],
                    "direction": "radial",
                    "opacity": 0.5
                },
                "class_type": "ImageGradientOverlay"
            },
            "4": {
                "inputs": {
                    "image": ["3", 0],
                    "shape_type": "geometric_grid",
                    "shapes": ["triangles", "hexagons", "wireframe_cubes"],
                    "animation": "rotate_3d",
                    "opacity": 0.6
                },
                "class_type": "ImageGeometricOverlay"
            },
            "5": {
                "inputs": {
                    "image": ["4", 0],
                    "edge_detection": True,
                    "edge_color": [255, 255, 255],
                    "edge_thickness": 2,
                    "neon_edges": True
                },
                "class_type": "ImageEdgeEnhance"
            },
            "6": {
                "inputs": {
                    "image": ["5", 0],
                    "hologram_effect": True,
                    "iridescence": 0.7,
                    "rainbow_shift": True,
                    "transparency_mode": "holographic"
                },
                "class_type": "ImageHologramEffect"
            },
            "7": {
                "inputs": {
                    "image": ["6", 0],
                    "text": "CHROME DREAMS",
                    "font": "futuristic_bold",
                    "effect": "chrome_3d",
                    "bevel": True,
                    "reflection": True
                },
                "class_type": "ImageChrome3DText"
            },
            "8": {
                "inputs": {
                    "image": ["7", 0],
                    "depth_of_field": True,
                    "focus_point": "center",
                    "blur_amount": 3,
                    "bokeh_effect": True
                },
                "class_type": "ImageDepthEffect"
            },
            "9": {
                "inputs": {
                    "images": ["8", 0],
                    "frame_rate": 30,
                    "format": "mp4",
                    "quality": "ultra_high",
                    "filename_prefix": "vaporwave_chrome_"
                },
                "class_type": "VHS_VideoOutput"
            }
        }
    }

    return workflow


def process_vaporwave_chrome(input_file, output_dir):
    """Process a single file with chrome vaporwave style"""

    print(f"🤖 CHROME VAPORWAVE: {input_file.name}")

    # ComfyUI processing command
    cmd = [
        sys.executable, "main.py",
        "--workflow", "vaporwave_chrome",
        "--input", str(input_file),
        "--output-dir", str(output_dir),
        "--extra-args", json.dumps({
            "metallic_intensity": 0.8,
            "geometric_complexity": 0.6,
            "hologram_effect": True,
            "chrome_reflection": 0.9
        })
    ]

    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Vaporwave Chrome Style Processing")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--text", default="CHROME DREAMS",
                        help="3D text overlay")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    result = process_vaporwave_chrome(input_file, output_dir)

    if result.returncode == 0:
        print("✅ CHROME VAPORWAVE SUCCESS!")
    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
