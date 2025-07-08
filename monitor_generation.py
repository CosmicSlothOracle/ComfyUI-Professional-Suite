#!/usr/bin/env python3
"""
Monitor ComfyUI Generation Progress
"""

import requests
import time
import os
from pathlib import Path


def monitor_generation():
    """Monitor the current generation and check for output"""
    server_url = "http://localhost:8188"
    prompt_id = "84b6ec4e-0cac-414d-bd12-bcd49dae48a1"

    print("🔍 Monitoring SDXL + LoRA Generation...")
    print(f"📋 Prompt ID: {prompt_id}")
    print("=" * 50)

    start_time = time.time()
    check_count = 0

    while True:
        check_count += 1
        elapsed = time.time() - start_time

        try:
            # Check queue status
            response = requests.get(f"{server_url}/queue")
            if response.status_code == 200:
                queue_data = response.json()
                running = queue_data.get('queue_running', [])
                pending = queue_data.get('queue_pending', [])

                # Check if our prompt is still running
                is_running = any(item[1] == prompt_id for item in running)
                is_pending = any(item[1] == prompt_id for item in pending)

                print(f"⏱️ Check #{check_count} - Elapsed: {elapsed:.1f}s")

                if is_running:
                    print("🔄 Still generating...")
                elif is_pending:
                    print("⏳ Waiting in queue...")
                else:
                    print("✅ Generation completed!")

                    # Check for output
                    check_output_files()
                    return True

            else:
                print(f"❌ Queue check failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Monitor error: {e}")

        # Wait before next check
        time.sleep(10)  # Check every 10 seconds

        # Timeout after 20 minutes
        if elapsed > 1200:
            print("⏰ Timeout after 20 minutes")
            return False


def check_output_files():
    """Check for generated output files"""
    output_dirs = [
        "ComfyUI_engine/output",
        "output",
        "ComfyUI_engine/user/default/output"
    ]

    print("\n📁 Checking for output files...")

    for output_dir in output_dirs:
        if Path(output_dir).exists():
            files = list(Path(output_dir).glob("*sdxl_lora_test*"))
            if files:
                print(f"✅ Found output in {output_dir}:")
                for file in files:
                    print(f"   📄 {file.name} ({file.stat().st_size} bytes)")
                return True

    print("🔍 No output files found yet")
    return False


def main():
    print("🔍 COMFYUI GENERATION MONITOR")
    print("=" * 40)

    success = monitor_generation()

    if success:
        print("\n🎉 Generation monitoring completed!")
    else:
        print("\n⏰ Monitoring timed out")


if __name__ == "__main__":
    main()
