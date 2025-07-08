#!/usr/bin/env python3
"""
VAPORWAVE TEST USING FFMPEG - REAL EFFECTS
Test vaporwave styles 1-3 using FFmpeg filters on specific user files
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# Specific test files provided by user
TEST_FILES = [
    "input/output-onlinegiftools (2)_fast_transparent.mp4",
    "input/2161e0c91f4e326f104ffe30552232ac_fast_transparent.mp4",
    "input/1938caaca4055d456a9c12ef8648a057_fast_transparent.mp4",
    "input/952fa268bac222d795de5a2729ac11d2_fast_transparent.mp4",
    "input/52de83165cfcec1ba2b2b49fe1c9d883_fast_transparent.mp4",
    "input/24ae4572aaab593bc1cc04383bc07591_fast_transparent.mp4",
    "input/23_fast_transparent.mp4"
]


def create_neon_effect(input_file, output_file):
    """Create neon vaporwave effect using FFmpeg"""

    # Neon effect: High saturation, pink/cyan color shift, glow
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-vf",
        (
            "eq=contrast=1.3:brightness=0.1:saturation=2.0,"  # High contrast/saturation
            "hue=h=300:s=1.5,"  # Shift to pink/magenta
            "curves=all='0/0 0.25/0.1 0.5/0.5 0.75/0.9 1/1',"  # S-curve
            "gblur=sigma=1:steps=2,"  # Slight glow
            "colorbalance=rs=0.3:gs=-0.2:bs=-0.1:rm=0.2:gm=-0.1:bm=-0.1"  # Pink/cyan balance
        ),
        "-c:a", "copy",  # Copy audio
        "-preset", "fast",
        str(output_file)
    ]

    return ffmpeg_cmd


def create_retro_effect(input_file, output_file):
    """Create retro sunset vaporwave effect using FFmpeg"""

    # Retro effect: Sunset colors, VHS degradation, warmth
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-vf",
        (
            "eq=contrast=1.2:saturation=1.8:gamma=0.9:brightness=0.1,"  # Retro color grading
            "hue=h=30:s=1.2,"  # Shift towards orange/red
            "colorbalance=rs=0.2:gs=0.1:bs=-0.2,"  # Warm sunset balance
            "noise=alls=3:allf=t,"  # VHS-like noise
            "unsharp=5:5:0.8:3:3:0.4"  # Slight unsharp for VHS look
        ),
        "-c:a", "copy",
        "-preset", "fast",
        str(output_file)
    ]

    return ffmpeg_cmd


def create_glitch_effect(input_file, output_file):
    """Create glitch cyberpunk vaporwave effect using FFmpeg"""

    # Glitch effect: Digital corruption, chromatic aberration, cyberpunk colors
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-vf",
        (
            "eq=contrast=1.5:saturation=2.2:brightness=0.2,"  # High contrast cyberpunk
            "hue=h=180:s=1.3,"  # Shift to cyan/blue
            "colorbalance=rs=-0.3:gs=-0.1:bs=0.4,"  # Cyan/blue cyberpunk balance
            "noise=alls=8:allf=t,"  # Digital noise
            "unsharp=7:7:1.5:7:7:0.8"  # Sharp digital look
        ),
        "-c:a", "copy",
        "-preset", "fast",
        str(output_file)
    ]

    return ffmpeg_cmd


def process_file_with_style(input_file, output_dir, style):
    """Process a single file with the specified vaporwave style"""

    input_path = Path(input_file)
    if not input_path.exists():
        return False, f"❌ File not found: {input_file}"

    output_file = output_dir / f"{input_path.stem}_{style}.mp4"

    print(f"🎨 {style.upper()}: Processing {input_path.name}...")

    try:
        start_time = time.time()

        # Create FFmpeg command based on style
        if style == "neon":
            cmd = create_neon_effect(input_path, output_file)
        elif style == "retro":
            cmd = create_retro_effect(input_path, output_file)
        elif style == "glitch":
            cmd = create_glitch_effect(input_path, output_file)
        else:
            return False, f"❌ Unknown style: {style}"

        # Run FFmpeg
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300)

        duration = time.time() - start_time

        if result.returncode == 0 and output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            return True, f"✅ {style.upper()}: {input_path.name} → {file_size:.1f}MB ({duration:.1f}s)"
        else:
            error_msg = result.stderr[:100] if result.stderr else "Unknown error"
            return False, f"❌ {style.upper()}: {input_path.name} → {error_msg}..."

    except subprocess.TimeoutExpired:
        return False, f"⏰ {style.upper()}: {input_path.name} → Timeout (>5min)"
    except Exception as e:
        return False, f"❌ {style.upper()}: {input_path.name} → {str(e)[:100]}..."


def test_vaporwave_ffmpeg():
    """Test vaporwave styles using FFmpeg on specific files"""

    print("🌈 VAPORWAVE FFMPEG TEST - STYLES 1-3")
    print("Testing: 🌈 NEON | 🌅 RETRO | ⚡ GLITCH")
    print("=" * 70)

    # Check FFmpeg availability
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg found and ready")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found! Please install FFmpeg first.")
        print("Download from: https://ffmpeg.org/download.html")
        return

    # Create test output directory
    test_dir = Path("output/vaporwave_test_ffmpeg")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Check which files exist
    existing_files = []
    missing_files = []

    for file_path in TEST_FILES:
        path = Path(file_path)
        if path.exists():
            existing_files.append(str(path))
        else:
            missing_files.append(file_path)

    if missing_files:
        print("⚠️  MISSING FILES:")
        for missing in missing_files:
            print(f"   {missing}")
        print()

    if not existing_files:
        print("❌ No test files found!")
        return

    print(f"✅ Found {len(existing_files)} files to process")
    print(f"📁 Output directory: {test_dir}")
    print("=" * 70)

    # Test each style
    styles = ["neon", "retro", "glitch"]
    results = {}
    total_start = time.time()

    for style in styles:
        if style == "neon":
            emoji, desc = "🌈", "Neon Aesthetics"
        elif style == "retro":
            emoji, desc = "🌅", "Retro Sunset"
        else:  # glitch
            emoji, desc = "⚡", "Glitch Cyberpunk"

        print(f"\n{emoji} TESTING {style.upper()} STYLE ({desc})")
        print("-" * 60)

        style_dir = test_dir / style
        style_dir.mkdir(exist_ok=True)

        style_results = []

        for file_path in existing_files:
            success, message = process_file_with_style(
                file_path, style_dir, style)
            style_results.append((success, message))
            print(f"   {message}")

        results[style] = style_results

        success_count = sum(1 for success, _ in style_results if success)
        print(
            f"   📊 {style.upper()}: {success_count}/{len(existing_files)} successful")

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("🎯 VAPORWAVE TEST RESULTS SUMMARY")
    print("=" * 70)

    overall_success = 0
    overall_total = 0

    for style in styles:
        style_results = results[style]
        success_count = sum(1 for success, _ in style_results if success)
        success_rate = success_count / len(style_results) * 100

        overall_success += success_count
        overall_total += len(style_results)

        if style == "neon":
            emoji, desc = "🌈", "Neon Aesthetics"
        elif style == "retro":
            emoji, desc = "🌅", "Retro Sunset"
        else:  # glitch
            emoji, desc = "⚡", "Glitch Cyberpunk"

        print(f"{emoji} {style.upper()} ({desc}):")
        print(
            f"   Success: {success_count}/{len(style_results)} files ({success_rate:.1f}%)")

        if success_count > 0:
            print(f"   ✅ Output: {test_dir / style}/")
        print()

    overall_rate = overall_success / overall_total * 100 if overall_total > 0 else 0
    print(
        f"📊 OVERALL: {overall_success}/{overall_total} files ({overall_rate:.1f}%)")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📁 All outputs: {test_dir}/")
    print("=" * 70)

    # Recommendations
    print("\n💡 NEXT STEPS:")
    if overall_success > 0:
        print("🎬 Review the generated videos in each style folder")
        print("🏆 Choose your preferred style for full batch processing")
        print("🚀 Use the chosen style with run_vaporwave_batch.py")
    else:
        print("❌ No files were processed successfully")
        print("🔧 Check FFmpeg installation and file paths")

    print(f"\n📂 Video outputs saved to: {test_dir}/")


def main():
    try:
        test_vaporwave_ffmpeg()
        print("\n🎉 Vaporwave test completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
