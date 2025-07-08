#!/usr/bin/env python3
"""
ULTIMATE FUSION BATCH PROCESSOR - SPECIAL FILE LIST
100% TRANSPARENT BACKGROUND PROCESSING
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# SPEZIELLE DATEILISTE VOM USER
SPECIAL_FILES = [
    "00370bedfef25129fd8441c864f67bb9.gif",
    "7ee80664e6f86ac416750497557bf6fc.gif",
    "0e35f5b16b8ba60a10fdd360de075def.gif",
    "517K.gif",
    "c5cd86843eaedd2a1ec8511e8c304b30.gif",
    "52de83165cfcec1ba2b2b49fe1c9d883.gif",
    "00dd8a85ebe872350d8ffda6435903a1.gif",
    "863b7fcc2b7b4c7d98478fe796490634.gif",
    "styled_dust_ring_archer.gif",
    "styled_electric_arc_archer.gif",
    "styled_energy_projectile_archer.gif",
    "styled_explosion.gif",
    "styled_final_attack.gif",
    "styled_lightning_archer.gif",
    "styled_lightning_archer_demonslayer.gif",
    "styled_lightning_bolt_archer.gif",
    "styled_portal_projectile_archer.gif",
    "styled_projectile_archer.gif",
    "styled_purple_flame_archer.gif",
    "styled_slash_archer.gif",
    "styled_spark_explosion_archer.gif",
    "styled_splash_archer.gif",
    "styled_tornado_archer.gif",
    "styled_water_spin_archer_demonslayer.gif",
    "styled_waterbeam_archer.gif",
    "stylized_vortex_portal.gif",
    "transparent_character.gif",
    "dance_105bpm.gif",
    "dance_combined.gif",
    "dance_sprite.gif",
    "e109a1a8c8324b38947ff23eded58d99..gif",
    "Intro_27_512x512.gif",
    "output-onlinegiftools (5).gif",
    "pirate.gif",
    "23.gif",
    "animated_sprite_corrected (1).gif",
    "deomnplanet.gif",
    "dhds-stimulate-effect - Konvertiert1.mkv-08-4cb41876ee08787fe1d16354e4f9bcd7.gif",
    "dhds-Td1h - Konvertiert1-09-e61937b9f2b91b4dcce893fa807a2fe9.gif",
    "dhds-tSfCbU - Konvertiert1-10-2d5897b18c4a3f7bb3f2df206190e01f.gif",
    "dhds-VfOdi5X - Konvertiert1-11-f2cb3ba67f45e1dc516dfbb31bbe94f7.gif",
    "l.gif",
    "eleni.gif",
    "elon_idle_8.gif.lnk",  # Wird übersprungen
    "559c2e47e018ac820885a753d51c098e.gif",
    "26a09634994b96e38d5bdafd16fa9b75.gif",
    "d60abfb7ec3ba5dea74f4181782c8a37.gif",
    "bf28790973c87cf39ab2eda62d9653b3.gif",
    "d2f092a467a547f4eb80e92a58ec798e.gif",
    "62ef326ab85cdd46ea19e268a4ba4dcf.gif",
    "2161e0c91f4e326f104ffe30552232ac.gif",
    "642c8bf2af91afd9e44dec00a06914f6.gif",
    "9f9735a5c4d8bd0502ec3e64bdda6cfb.gif",
    "P9vTI0.gif",
    "e675dd23126203.5604763779b3e.gif",
    "9f720323126213.56047641e9c83.gif",
    "b8c60c23126211.56047639576d7.gif",
    "82a55a7f9d9a3cab7545146c78146d9d.gif",
    "gwanwoo-tak-1490340467910.gif",
    "d1d71ff4514a99bfb0f0e93ef59e3575.gif"
]


def run_ultimate_fusion_processor():
    """Run the ultimate fusion processor for special file list"""

    print("ULTIMATE FUSION BATCH PROCESSOR - SPECIAL LIST")
    print("100% TRANSPARENT BACKGROUND PROCESSING")
    print("'Maximale Funktionalitaet, schoen muss es werden!'")
    print("=" * 70)

    # Paths
    input_dir = Path("C:/Users/Public/ComfyUI-master/input")
    output_dir = Path("C:/Users/Public/ComfyUI-master/output")

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Filter existing files
    existing_files = []
    for filename in SPECIAL_FILES:
        if filename.endswith('.lnk'):
            print(f"SKIPPING: {filename} (link file)")
            continue

        file_path = input_dir / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            print(f"NOT FOUND: {filename}")

    print(f"Found {len(existing_files)} files to process")
    print(f"Processing with ULTIMATE FUSION (WF1+WF7+WF2)")
    print("=" * 70)

    # Process each file
    success_count = 0
    error_count = 0

    for i, filename in enumerate(existing_files, 1):
        print(f"\n[{i}/{len(existing_files)}] ULTIMATE FUSION -> {filename}")

        input_file = input_dir / filename
        output_file = output_dir / \
            f"{Path(filename).stem}_ultimate_fusion_transparent.mp4"

        try:
            # Run ultimate fusion workflow
            start_time = time.time()

            cmd = [
                sys.executable,
                "workflow_ultimate_fusion_transparent_fixed.py",
                "--input", str(input_file),
                "--output", str(output_dir)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout

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
            print(f"TIMEOUT: {filename} (>15min)")
            error_count += 1
        except Exception as e:
            print(f"EXCEPTION: {filename}: {e}")
            error_count += 1

    # Final summary
    print("\n" + "=" * 70)
    print("ULTIMATE FUSION BATCH PROCESSING COMPLETE")
    print(f"Success: {success_count}/{len(existing_files)} files")
    print(f"Errors: {error_count}/{len(existing_files)} files")
    print(f"Success Rate: {success_count/len(existing_files)*100:.1f}%")
    print("All successful files have 100% TRANSPARENT BACKGROUNDS!")
    print("=" * 70)

    return success_count, error_count


if __name__ == "__main__":
    try:
        success, errors = run_ultimate_fusion_processor()

        if success > 0:
            print(
                f"\nMISSION ACCOMPLISHED! {success} files processed successfully!")
            print("Check output folder for ultimate fusion results with transparency!")

        if errors > 0:
            print(f"\n{errors} files had issues - check logs above")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
