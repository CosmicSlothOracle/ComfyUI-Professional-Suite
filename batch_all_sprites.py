#!/usr/bin/env python3
"""
BATCH ALL SPRITES PROCESSOR
Verarbeitet alle Spritesheets im input Ordner automatisch
"""

import os
import sys
from pathlib import Path
import time
from standalone_enhanced_spritesheet import StandaloneEnhancedSpritesheetProcessor


def main():
    print("🎮 BATCH ALL SPRITES PROCESSOR")
    print("=" * 60)

    # Initialize processor
    processor = StandaloneEnhancedSpritesheetProcessor()

    # Input directory
    input_dir = Path("input")
    output_base = Path("output/batch_sprites_processed")
    output_base.mkdir(parents=True, exist_ok=True)

    # Find all sprite files
    sprite_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        sprite_files.extend(input_dir.glob(ext))

    # Also check sprite_sheets subdirectory
    sprite_sheets_dir = input_dir / "sprite_sheets"
    if sprite_sheets_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(sprite_sheets_dir.glob(ext))

    # Filter by size (only large images, likely spritesheets)
    valid_files = []
    for file in sprite_files:
        try:
            size = file.stat().st_size
            if size > 100_000:  # > 100KB
                valid_files.append(file)
                print(f"📋 Found: {file.name} ({size // 1024}KB)")
        except:
            continue

    if not valid_files:
        print("❌ No spritesheet files found!")
        return

    print(f"\n📊 Processing {len(valid_files)} files...")

    processed_count = 0
    failed_count = 0

    for i, sprite_file in enumerate(valid_files, 1):
        print(f"\n[{i}/{len(valid_files)}] " + "="*40)
        print(f"🎮 Processing: {sprite_file.name}")

        start_time = time.time()

        try:
            # Create output directory for this sprite
            sprite_output = output_base / sprite_file.stem
            sprite_output.mkdir(parents=True, exist_ok=True)

            # Process spritesheet
            result = processor.process_spritesheet_enhanced(
                sprite_file,
                adaptive_tolerance=True,
                tolerance_override=35.0,
                aggressive_extraction=True,
                min_area_factor=2500.0,
                edge_refinement=True,
                hsv_analysis=True,
                multi_zone_sampling=True,
                morphological_cleanup=True,
                smooth_edges=True
            )

            # Save results to output directory
            if result['extracted_frames']:
                for j, frame in enumerate(result['extracted_frames']):
                    frame_path = sprite_output / f"frame_{j:03d}.png"
                    # Convert numpy array to PIL Image and save
                    from PIL import Image
                    if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                        pil_frame = Image.fromarray(frame, 'RGBA')
                    else:  # RGB
                        pil_frame = Image.fromarray(frame, 'RGB')
                    pil_frame.save(frame_path)

                # Save processing info
                info_path = sprite_output / "processing_info.txt"
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(result['processing_info'])
                    f.write(f"\nFrame Statistics:\n{result['frame_stats']}")

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"✅ Success: {len(result['extracted_frames'])} frames")
            print(f"⏱️ Time: {processing_time:.2f}s")
            print(f"📁 Output: {sprite_output}")

            processed_count += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            failed_count += 1

        # Small delay between files
        time.sleep(0.5)

    # Final statistics
    print("\n" + "="*60)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total files: {len(valid_files)}")
    print(f"💯 Success rate: {(processed_count / len(valid_files) * 100):.1f}%")
    print(f"📁 All outputs saved to: {output_base.absolute()}")


if __name__ == "__main__":
    main()
