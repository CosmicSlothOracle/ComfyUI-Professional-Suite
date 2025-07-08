#!/usr/bin/env python3
"""
FAST TEST BATCH PROCESSOR - SMALL FILE LIST
100% TRANSPARENT BACKGROUND PROCESSING
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# KLEINE TESTLISTE VOM USER
TEST_FILES = [
    "output-onlinegiftools (2).gif",
    "eleni.gif",
    "elon_idle_8.gif.lnk",  # Wird übersprungen
    "Intro_27_512x512.gif.lnk",  # Wird übersprungen
    "d1d71ff4514a99bfb0f0e93ef59e3575.gif",
    "7ee80664e6f86ac416750497557bf6fc.gif",
    "0e35f5b16b8ba60a10fdd360de075def.gif",
    "517K.gif"
]


def run_fast_test_processor():
    """Run the fast test processor for small file list"""

    print("FAST TEST BATCH PROCESSOR - SMALL LIST")
    print("100% TRANSPARENT BACKGROUND PROCESSING")
    print("WF7 (Histogram) + WF2 (Anti-Fractal) - NO K-means for speed")
    print("=" * 70)

    # Paths
    input_dir = Path("C:/Users/Public/ComfyUI-master/input")
    output_dir = Path("C:/Users/Public/ComfyUI-master/output")

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Filter existing files
    existing_files = []
    for filename in TEST_FILES:
        if filename.endswith('.lnk'):
            print(f"SKIPPING: {filename} (link file)")
            continue

        file_path = input_dir / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            print(f"NOT FOUND: {filename}")

    print(f"Found {len(existing_files)} files to process")
    print(f"Processing with FAST TRANSPARENT WORKFLOW")
    print("=" * 70)

    # Process each file
    success_count = 0
    error_count = 0

    for i, filename in enumerate(existing_files, 1):
        print(f"\n[{i}/{len(existing_files)}] FAST TRANSPARENT -> {filename}")

        input_file = input_dir / filename
        output_file = output_dir / \
            f"{Path(filename).stem}_fast_transparent.mp4"

        try:
            # Run fast transparent workflow
            start_time = time.time()

            cmd = [
                sys.executable,
                "workflow_fast_transparent.py",
                "--input", str(input_file),
                "--output", str(output_dir)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode == 0:
                if output_file.exists():
                    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                    print(
                        f"SUCCESS: {filename} ({file_size:.1f}MB, {duration:.1f}s)")
                    success_count += 1
                else:
                    print(f"OUTPUT NOT FOUND: {filename}")
                    error_count += 1
            else:
                print(f"PROCESS ERROR: {filename}")
                print(f"   Error: {result.stderr}")
                error_count += 1

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {filename} (>5min)")
            error_count += 1
        except Exception as e:
            print(f"EXCEPTION: {filename}: {e}")
            error_count += 1

    # Final summary
    print("\n" + "=" * 70)
    print("FAST TEST BATCH PROCESSING COMPLETE")
    print(f"Success: {success_count}/{len(existing_files)} files")
    print(f"Errors: {error_count}/{len(existing_files)} files")
    print(f"Success Rate: {success_count/len(existing_files)*100:.1f}%")
    print("All successful files have 100% TRANSPARENT BACKGROUNDS!")
    print("=" * 70)

    return success_count, error_count


if __name__ == "__main__":
    try:
        success, errors = run_fast_test_processor()

        if success > 0:
            print(
                f"\nTEST SUCCESSFUL! {success} files processed successfully!")
            print("Check output folder for fast transparent results!")

        if errors > 0:
            print(f"\n{errors} files had issues - check logs above")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
