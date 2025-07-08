#!/usr/bin/env python3
"""
Batch processor for additional spritesheets - Second batch
Processes 400+ new ChatGPT and German-named sprite files
"""

from intelligent_spritesheet_processor import IntelligentSpritesheetProcessor
import os
import sys
import time
from pathlib import Path

# Add ComfyUI path
sys.path.insert(0, str(Path(__file__).parent))

# Import our processor


def main():
    # New batch file list (shortened for processing)
    new_files = [
        "ChatGPT Image 29. Juni 2025, 10_28_31.png",
        "ChatGPT Image 29. Juni 2025, 10_28_33.png",
        "ChatGPT Image 29. Juni 2025, 10_27_08.png",
        "ChatGPT Image 29. Juni 2025, 10_27_17.png",
        "ChatGPT Image 29. Juni 2025, 10_27_06.png",
        "ChatGPT Image 29. Juni 2025, 10_27_07.png",
        "ChatGPT Image 29. Juni 2025, 10_27_04.png",
        "ChatGPT Image 29. Juni 2025, 10_27_12.png",
        "ChatGPT Image 29. Juni 2025, 10_26_52.png",
        "ChatGPT Image 29. Juni 2025, 10_26_55.png",
        # ... (Full list would be processed - condensed for efficiency)
        "Tanzbewegungen eines jungen Charakters.png",
        "ChatGPT Image 29. Juni 2025, 10_30_06.png"
    ]

    # Base paths
    input_dir = Path("input")
    output_base = Path("output/spritesheet_batch_2")

    # Create processor
    processor = IntelligentSpritesheetProcessor()

    # Process all files that exist
    results = []
    start_time = time.time()

    existing_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') and ('ChatGPT Image 29. Juni 2025' in filename or
                                          any(german_word in filename for german_word in
                                              ['Tanzbewegungen', 'Charakter', 'Mann', 'Frau', 'Kampf'])):
            existing_files.append(filename)

    print(f"Found {len(existing_files)} files to process")

    for i, filename in enumerate(existing_files, 1):
        try:
            input_path = input_dir / filename
            output_dir = output_base / filename.replace('.png', '')

            print(f"[{i}/{len(existing_files)}] Processing: {filename}")

            result = processor.process_spritesheet(
                str(input_path),
                bg_removal_method="corner_detection"
            )

            if 'error' not in result:
                print(f"  ✓ Extracted {result['total_frames']} frames")
                results.append(result)
            else:
                print(f"  ✗ Failed: {result['error']}")

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    # Summary
    elapsed = time.time() - start_time
    total_frames = sum(r['total_frames'] for r in results)

    print(f"\n=== BATCH 2 PROCESSING COMPLETE ===")
    print(f"Files processed: {len(results)}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Average frames per spritesheet: {total_frames/len(results):.1f}")
    print(f"Processing speed: {total_frames/elapsed:.1f} frames/second")


if __name__ == "__main__":
    main()
