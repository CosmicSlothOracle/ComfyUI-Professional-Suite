#!/usr/bin/env python3
"""
VAPORWAVE BATCH PROCESSOR - MASTER CONTROLLER
Apply various vaporwave styles to all transparent background videos
Choose from: neon, retro, glitch, chrome styles
"""

import subprocess
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Vaporwave style options
VAPORWAVE_STYLES = {
    "neon": {
        "script": "workflow_vaporwave_neon.py",
        "description": "🌈 Neon Aesthetics - Bright neon colors, glowing effects, grid overlays",
        "prefix": "vaporwave_neon_"
    },
    "retro": {
        "script": "workflow_vaporwave_retro.py",
        "description": "🌅 Retro Sunset - Palm trees, sunset gradients, VHS effects",
        "prefix": "vaporwave_retro_"
    },
    "glitch": {
        "script": "workflow_vaporwave_glitch.py",
        "description": "⚡ Glitch Cyberpunk - Digital corruption, datamoshing, chromatic aberration",
        "prefix": "vaporwave_glitch_"
    },
    "chrome": {
        "script": "workflow_vaporwave_chrome.py",
        "description": "🤖 Chrome Geometry - Metallic surfaces, 3D shapes, holographic effects",
        "prefix": "vaporwave_chrome_"
    }
}


def get_transparent_videos(input_dir):
    """Get all transparent background videos from output directory"""

    transparent_videos = []
    output_dir = Path(input_dir)

    # Look for fast_transparent.mp4 files
    for video_file in output_dir.glob("*_fast_transparent.mp4"):
        transparent_videos.append(video_file)

    return sorted(transparent_videos)


def process_single_video(video_file, style, output_dir, custom_text=None):
    """Process a single video with chosen vaporwave style"""

    style_info = VAPORWAVE_STYLES[style]
    script_path = Path(style_info["script"])

    if not script_path.exists():
        return f"❌ Script not found: {script_path}"

    # Build command
    cmd = [
        sys.executable, str(script_path),
        "--input", str(video_file),
        "--output", str(output_dir)
    ]

    # Add custom text for applicable styles
    if custom_text and style in ["retro", "chrome"]:
        cmd.extend(["--text", custom_text])

    # Add glitch intensity for glitch style
    if style == "glitch":
        cmd.extend(["--glitch-intensity", "0.6"])

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True,
                                text=True, timeout=600)  # 10 min timeout
        duration = time.time() - start_time

        if result.returncode == 0:
            # Check if output file was created
            expected_output = output_dir / \
                f"{video_file.stem.replace('_fast_transparent', '')}_{style_info['prefix']}.mp4"
            if expected_output.exists():
                file_size = expected_output.stat().st_size / (1024 * 1024)  # MB
                return f"✅ {video_file.name} -> {style.upper()} ({file_size:.1f}MB, {duration:.1f}s)"
            else:
                return f"❌ {video_file.name} -> Output file not created"
        else:
            return f"❌ {video_file.name} -> Process error: {result.stderr[:100]}..."

    except subprocess.TimeoutExpired:
        return f"⏰ {video_file.name} -> Timeout (>10min)"
    except Exception as e:
        return f"❌ {video_file.name} -> Exception: {str(e)[:100]}..."


def run_vaporwave_batch(style, max_workers=3, custom_text=None, file_limit=None):
    """Run vaporwave batch processing"""

    print("🌈 VAPORWAVE BATCH PROCESSOR")
    print("=" * 60)

    # Validate style
    if style not in VAPORWAVE_STYLES:
        print(f"❌ Invalid style: {style}")
        print(f"Available styles: {', '.join(VAPORWAVE_STYLES.keys())}")
        return False

    style_info = VAPORWAVE_STYLES[style]
    print(f"Style: {style_info['description']}")
    print("=" * 60)

    # Setup directories
    input_dir = Path("output")  # Transparent videos are in output folder
    vaporwave_dir = Path("output/vaporwave") / style
    vaporwave_dir.mkdir(parents=True, exist_ok=True)

    # Get all transparent videos
    transparent_videos = get_transparent_videos(input_dir)

    if not transparent_videos:
        print("❌ No transparent videos found in output directory!")
        print("Run the transparent background generation first.")
        return False

    # Apply file limit if specified
    if file_limit:
        transparent_videos = transparent_videos[:file_limit]

    print(f"Found {len(transparent_videos)} transparent videos to process")
    print(f"Output directory: {vaporwave_dir}")
    print(f"Max workers: {max_workers}")
    print("=" * 60)

    # Process videos
    success_count = 0
    error_count = 0
    start_total = time.time()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_single_video, video, style, vaporwave_dir, custom_text): video
            for video in transparent_videos
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_video), 1):
            video = future_to_video[future]

            try:
                result = future.result()
                print(f"[{i}/{len(transparent_videos)}] {result}")

                if result.startswith("✅"):
                    success_count += 1
                else:
                    error_count += 1

            except Exception as e:
                print(
                    f"[{i}/{len(transparent_videos)}] ❌ {video.name} -> Exception: {e}")
                error_count += 1

            # Progress update every 10 files
            if i % 10 == 0:
                elapsed = time.time() - start_total
                avg_time = elapsed / i
                remaining = (len(transparent_videos) - i) * avg_time
                print(f"\n--- PROGRESS UPDATE ---")
                print(
                    f"Processed: {i}/{len(transparent_videos)} ({i/len(transparent_videos)*100:.1f}%)")
                print(f"Success: {success_count}, Errors: {error_count}")
                print(f"ETA: {remaining/60:.1f} minutes")
                print("=" * 30)

    # Final summary
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print("VAPORWAVE BATCH PROCESSING COMPLETE")
    print(f"Style: {style.upper()} - {style_info['description']}")
    print(f"Success: {success_count}/{len(transparent_videos)} files")
    print(f"Errors: {error_count}/{len(transparent_videos)} files")
    print(f"Success Rate: {success_count/len(transparent_videos)*100:.1f}%")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(transparent_videos):.1f}s per file")
    print(f"Output Location: {vaporwave_dir}")
    print("=" * 60)

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="Vaporwave Batch Processor")
    parser.add_argument("--style", required=True, choices=list(VAPORWAVE_STYLES.keys()),
                        help="Vaporwave style to apply")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers")
    parser.add_argument(
        "--text", help="Custom text overlay (for retro/chrome styles)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to process (for testing)")
    parser.add_argument("--list-styles", action="store_true",
                        help="List available styles")

    args = parser.parse_args()

    if args.list_styles:
        print("Available Vaporwave Styles:")
        print("=" * 40)
        for style, info in VAPORWAVE_STYLES.items():
            print(f"{style}: {info['description']}")
        return

    try:
        success = run_vaporwave_batch(
            style=args.style,
            max_workers=args.workers,
            custom_text=args.text,
            file_limit=args.limit
        )

        if success:
            print(
                f"\n🎉 VAPORWAVE SUCCESS! Check output/vaporwave/{args.style}/ folder!")
        else:
            print(f"\n❌ VAPORWAVE FAILED! Check error messages above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
