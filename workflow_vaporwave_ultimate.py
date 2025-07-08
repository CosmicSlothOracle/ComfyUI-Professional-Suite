#!/usr/bin/env python3
"""
ULTIMATE VAPORWAVE WORKFLOW - REAL FFMPEG EFFECTS
Advanced vaporwave transformation with authentic 80s/90s aesthetics
Combines multiple visual effects for true vaporwave experience
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path
import tempfile


def create_neon_synthwave_effect(input_file, output_file):
    """Create authentic neon synthwave vaporwave effect"""

    # Create temporary files for multi-pass processing
    temp_dir = Path(tempfile.gettempdir())
    temp1 = temp_dir / f"vw_temp1_{os.getpid()}.mp4"
    temp2 = temp_dir / f"vw_temp2_{os.getpid()}.mp4"

    try:
        # PASS 1: Base color grading and neon color shift
        cmd1 = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vf", (
                "eq=contrast=1.4:brightness=0.2:saturation=2.5:gamma=0.8,"  # High contrast neon base
                "hue=h=300:s=2.0,"  # Strong magenta/pink shift
                "curves=all='0/0.05 0.1/0.15 0.3/0.4 0.7/0.85 1/0.95',"  # Neon curve
                "colorbalance=rs=0.4:gs=-0.3:bs=0.2:rm=0.3:gm=-0.2:bm=0.1"  # Pink/cyan balance
            ),
            "-c:a", "copy", "-preset", "fast", str(temp1)
        ]

        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            raise Exception(f"Pass 1 failed: {result1.stderr}")

        # PASS 2: Add glow and grid overlay effects
        cmd2 = [
            "ffmpeg", "-y", "-i", str(temp1),
            "-vf", (
                "gblur=sigma=2:steps=3,"  # Glow effect
                "unsharp=5:5:1.5:5:5:0.5,"  # Enhance edges for neon look
                # Grid overlay using overlay filter with generated grid
                f"drawgrid=width=32:height=32:thickness=1:color=magenta@0.3,"
                "noise=alls=2:allf=t+u"  # Subtle VHS-like noise
            ),
            "-c:a", "copy", "-preset", "fast", str(output_file)
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise Exception(f"Pass 2 failed: {result2.stderr}")

        return True, "✅ Neon Synthwave effect applied successfully"

    except Exception as e:
        return False, f"❌ Neon effect failed: {str(e)}"
    finally:
        # Cleanup temp files
        for temp_file in [temp1, temp2]:
            if temp_file.exists():
                temp_file.unlink()


def create_retro_sunset_effect(input_file, output_file):
    """Create authentic retro sunset vaporwave effect"""

    temp_dir = Path(tempfile.gettempdir())
    temp1 = temp_dir / f"vw_temp1_{os.getpid()}.mp4"

    try:
        # PASS 1: Sunset color grading
        cmd1 = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vf", (
                "eq=contrast=1.3:brightness=0.15:saturation=2.0:gamma=0.9,"  # Warm retro base
                "hue=h=20:s=1.8,"  # Orange/pink sunset shift
                "curves=r='0/0.1 0.5/0.6 1/0.9':g='0/0.05 0.5/0.5 1/0.8':b='0/0 0.5/0.3 1/0.7',"  # Sunset curve
                "colorbalance=rs=0.3:gs=0.1:bs=-0.3:rm=0.2:gm=0.05:bm=-0.2"  # Warm sunset balance
            ),
            "-c:a", "copy", "-preset", "fast", str(temp1)
        ]

        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            raise Exception(f"Pass 1 failed: {result1.stderr}")

        # PASS 2: VHS degradation and retro effects
        cmd2 = [
            "ffmpeg", "-y", "-i", str(temp1),
            "-vf", (
                "noise=alls=8:allf=t,"  # VHS tape noise
                "unsharp=3:3:0.8:3:3:0.4,"  # VHS softness
                # Add horizontal scanlines for retro TV effect
                "format=yuv420p,"
                # Scanlines
                "geq=lum='if(mod(Y,4),lum(X,Y),lum(X,Y)*0.8)':cb=cb:cr=cr,"
                "hqx=3"  # Pixel art upscaling for retro look
            ),
            "-c:a", "copy", "-preset", "fast", str(output_file)
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise Exception(f"Pass 2 failed: {result2.stderr}")

        return True, "✅ Retro Sunset effect applied successfully"

    except Exception as e:
        return False, f"❌ Retro effect failed: {str(e)}"
    finally:
        if temp1.exists():
            temp1.unlink()


def create_glitch_cyberpunk_effect(input_file, output_file):
    """Create authentic glitch cyberpunk vaporwave effect"""

    temp_dir = Path(tempfile.gettempdir())
    temp1 = temp_dir / f"vw_temp1_{os.getpid()}.mp4"

    try:
        # PASS 1: Cyberpunk color grading and digital corruption
        cmd1 = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vf", (
                "eq=contrast=1.6:brightness=0.1:saturation=2.3:gamma=0.7,"  # High contrast digital
                "hue=h=180:s=2.2,"  # Cyan/blue cyberpunk shift
                "curves=all='0/0.1 0.2/0.05 0.4/0.3 0.6/0.7 0.8/0.95 1/1',"  # Digital curve
                "colorbalance=rs=-0.4:gs=-0.1:bs=0.5:rm=-0.3:gm=0:bm=0.4"  # Cyan cyberpunk balance
            ),
            "-c:a", "copy", "-preset", "fast", str(temp1)
        ]

        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            raise Exception(f"Pass 1 failed: {result1.stderr}")

        # PASS 2: Digital glitch effects
        cmd2 = [
            "ffmpeg", "-y", "-i", str(temp1),
            "-vf", (
                "noise=alls=15:allf=t+u,"  # Heavy digital noise
                "unsharp=7:7:2.0:7:7:1.0,"  # Sharp digital edges
                # Chromatic aberration simulation
                "split[main][dup];"
                "[dup]hue=h=10,crop=iw-6:ih:3:0[r];"
                "[main][r]overlay=0:0:format=auto,format=yuv420p,"
                # Add digital corruption patterns
                # Random pixel corruption
                "geq=lum='if(gt(random(0)*255,240),255,lum(X,Y))':cb=cb:cr=cr"
            ),
            "-c:a", "copy", "-preset", "fast", str(output_file)
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise Exception(f"Pass 2 failed: {result2.stderr}")

        return True, "✅ Glitch Cyberpunk effect applied successfully"

    except Exception as e:
        return False, f"❌ Glitch effect failed: {str(e)}"
    finally:
        if temp1.exists():
            temp1.unlink()


def process_vaporwave_ultimate(input_file, output_dir, style="neon"):
    """Process video with ultimate vaporwave effects"""

    input_path = Path(input_file)
    if not input_path.exists():
        return False, f"❌ Input file not found: {input_file}"

    # Create output filename
    output_file = output_dir / f"{input_path.stem}_vaporwave_{style}.mp4"

    print(f"🌈 ULTIMATE VAPORWAVE ({style.upper()}): {input_path.name}")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")

    # Apply the selected effect
    if style.lower() == "neon":
        success, message = create_neon_synthwave_effect(
            input_path, output_file)
    elif style.lower() == "retro":
        success, message = create_retro_sunset_effect(input_path, output_file)
    elif style.lower() == "glitch":
        success, message = create_glitch_cyberpunk_effect(
            input_path, output_file)
    else:
        return False, f"❌ Unknown style: {style}. Use: neon, retro, glitch"

    return success, message


def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Vaporwave Processing with Real FFmpeg Effects")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--style", choices=["neon", "retro", "glitch"], default="neon",
                        help="Vaporwave style to apply")

    args = parser.parse_args()

    # Check FFmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found! Please install FFmpeg first.")
        print("Download from: https://ffmpeg.org/download.html")
        sys.exit(1)

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    success, message = process_vaporwave_ultimate(
        input_file, output_dir, args.style)

    print(f"\n{message}")

    if not success:
        sys.exit(1)

    print(f"\n✅ ULTIMATE VAPORWAVE SUCCESS!")
    print(f"📁 Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
