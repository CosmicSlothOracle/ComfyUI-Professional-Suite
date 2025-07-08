#!/usr/bin/env python3
"""
Pixel Art Pipeline Test Script
Tests 4 specific GIF files before full batch processing
"""

import json
import requests
import time
import os
from pathlib import Path
import uuid


class PixelArtPipelineTester:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8188"
        self.base_dir = Path("C:/Users/Public/ComfyUI-master")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output" / "pixel_art_test"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test files specified by user
        self.test_files = [
            "c0a420e57c75f1f5863d48197fd19c3a_fast_transparent_converted.gif",
            "eleni_fast_transparent_converted.gif",
            "0e35f5b16b8ba60a10fdd360de075def_fast_transparent_converted.gif",
            "9f720323126213.56047641e9c83_fast_transparent_converted.gif"
        ]

    def wait_for_comfyui(self, max_wait=120):
        """Wait for ComfyUI to be ready"""
        print("🔄 Waiting for ComfyUI to start...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.base_url}/system_stats", timeout=5)
                if response.status_code == 200:
                    print("✅ ComfyUI is ready!")
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            print(f"   ⏳ Waiting... ({int(time.time() - start_time)}s)")
            time.sleep(5)

        print("❌ ComfyUI failed to start within timeout")
        return False

    def create_test_workflow(self, gif_filename):
        """Create optimized workflow for test file"""
        workflow = {
            "last_node_id": 8,
            "last_link_id": 10,
            "nodes": [
                {
                    "id": 1,
                    "type": "LoadVideo",
                    "pos": [50, 100],
                    "size": {"0": 315, "1": 58},
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {
                            "name": "VIDEO",
                            "type": "VIDEO",
                            "links": [1],
                            "slot_index": 0
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "LoadVideo"
                    },
                    "widgets_values": [gif_filename]
                },
                {
                    "id": 2,
                    "type": "GetVideoComponents",
                    "pos": [400, 100],
                    "size": {"0": 315, "1": 106},
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "video",
                            "type": "VIDEO",
                            "link": 1
                        }
                    ],
                    "outputs": [
                        {
                            "name": "IMAGES",
                            "type": "IMAGE",
                            "links": [2],
                            "slot_index": 0
                        },
                        {
                            "name": "fps",
                            "type": "FLOAT",
                            "links": [3],
                            "slot_index": 1
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "GetVideoComponents"
                    }
                },
                {
                    "id": 3,
                    "type": "PixelArtDetector",
                    "pos": [750, 100],
                    "size": {"0": 400, "1": 600},
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "image",
                            "type": "IMAGE",
                            "link": 2
                        }
                    ],
                    "outputs": [
                        {
                            "name": "IMAGE",
                            "type": "IMAGE",
                            "links": [4],
                            "slot_index": 0
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "PixelArtDetector"
                    },
                    "widgets_values": [
                        "gameboy",      # palette
                        "Image.quantize",  # method
                        1,              # scale_factor
                        128,            # target_width
                        128,            # target_height
                        True,           # maintain_aspect_ratio
                        16,             # colors
                        True,           # enable_dithering
                        8,              # dither_strength
                        "MAXCOVERAGE",  # palette_method
                        "KMEANS_PP_CENTERS",  # kmeans_init
                        3,              # kmeans_iter
                        50,             # max_iterations
                        True,           # optimize_palette
                        0.02,           # color_threshold
                        True,           # preserve_transparency
                        "floyd_steinberg"  # dithering_method
                    ]
                },
                {
                    "id": 4,
                    "type": "CreateVideo",
                    "pos": [1200, 100],
                    "size": {"0": 315, "1": 178},
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "images",
                            "type": "IMAGE",
                            "link": 4
                        },
                        {
                            "name": "fps",
                            "type": "FLOAT",
                            "link": 3
                        }
                    ],
                    "outputs": [
                        {
                            "name": "VIDEO",
                            "type": "VIDEO",
                            "links": [5],
                            "slot_index": 0
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "CreateVideo"
                    },
                    "widgets_values": [
                        24,    # fps override
                        0.8,   # quality
                        "h264"  # codec
                    ]
                },
                {
                    "id": 5,
                    "type": "SaveVideo",
                    "pos": [1550, 100],
                    "size": {"0": 315, "1": 58},
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "video",
                            "type": "VIDEO",
                            "link": 5
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "SaveVideo"
                    },
                    "widgets_values": [
                        f"pixel_art_test_{gif_filename.replace('.gif', '.mp4')}"
                    ]
                }
            ],
            "links": [
                [1, 1, 0, 2, 0, "VIDEO"],
                [2, 2, 0, 3, 0, "IMAGE"],
                [3, 2, 1, 4, 1, "FLOAT"],
                [4, 3, 0, 4, 0, "IMAGE"],
                [5, 4, 0, 5, 0, "VIDEO"]
            ],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4
        }

        return workflow

    def submit_workflow(self, workflow_data):
        """Submit workflow to ComfyUI"""
        try:
            client_id = str(uuid.uuid4())

            response = requests.post(
                f"{self.base_url}/prompt",
                json={
                    "prompt": workflow_data,
                    "client_id": client_id
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("prompt_id"), client_id
            else:
                print(f"❌ Failed to submit workflow: {response.status_code}")
                print(f"Response: {response.text}")
                return None, None

        except Exception as e:
            print(f"❌ Error submitting workflow: {e}")
            return None, None

    def check_workflow_status(self, prompt_id):
        """Check workflow completion status"""
        try:
            # Check history first
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}", timeout=10)
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    return "completed"

            # Check queue
            response = requests.get(f"{self.base_url}/queue", timeout=10)
            if response.status_code == 200:
                queue_data = response.json()

                # Check running queue
                for item in queue_data.get("queue_running", []):
                    if len(item) > 1 and item[1] == prompt_id:
                        return "running"

                # Check pending queue
                for item in queue_data.get("queue_pending", []):
                    if len(item) > 1 and item[1] == prompt_id:
                        return "pending"

            return "unknown"

        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return "error"

    def test_single_file(self, gif_filename):
        """Test processing of a single GIF file"""
        print(f"\n🎨 Testing: {gif_filename}")

        # Check if input file exists
        input_file = self.input_dir / gif_filename
        if not input_file.exists():
            print(f"   ❌ Input file not found: {input_file}")
            return False

        print(f"   📁 Input file found: {input_file.stat().st_size} bytes")

        try:
            # Create workflow
            workflow = self.create_test_workflow(gif_filename)

            # Submit workflow
            prompt_id, client_id = self.submit_workflow(workflow)
            if not prompt_id:
                return False

            print(f"   📤 Submitted workflow (ID: {prompt_id[:8]}...)")

            # Wait for completion
            max_wait = 600  # 10 minutes per test file
            start_time = time.time()
            last_status = ""

            while time.time() - start_time < max_wait:
                status = self.check_workflow_status(prompt_id)

                if status != last_status:
                    print(f"   📊 Status: {status}")
                    last_status = status

                if status == "completed":
                    elapsed = int(time.time() - start_time)
                    print(f"   ✅ Completed successfully in {elapsed}s")
                    return True
                elif status == "error":
                    print(f"   ❌ Failed with error")
                    return False

                time.sleep(10)

            print(f"   ⏰ Timeout after {max_wait}s")
            return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def run_test_pipeline(self):
        """Run the complete test pipeline"""
        print("🚀 STARTING PIXEL ART PIPELINE TEST")
        print("=" * 50)

        # Wait for ComfyUI
        if not self.wait_for_comfyui():
            print("❌ ComfyUI not available - cannot run tests")
            return

        print(f"📊 Testing {len(self.test_files)} files:")
        for i, filename in enumerate(self.test_files, 1):
            print(f"   {i}. {filename}")

        # Test each file
        results = {}
        successful = 0

        for i, filename in enumerate(self.test_files, 1):
            print(f"\n[{i}/{len(self.test_files)}] Processing test file...")

            if self.test_single_file(filename):
                results[filename] = "SUCCESS"
                successful += 1
            else:
                results[filename] = "FAILED"

        # Final report
        print("\n" + "=" * 50)
        print("🎯 TEST PIPELINE RESULTS")
        print(
            f"✅ Successful: {successful}/{len(self.test_files)} ({successful/len(self.test_files)*100:.1f}%)")
        print(
            f"❌ Failed: {len(self.test_files)-successful}/{len(self.test_files)}")

        print("\n📋 DETAILED RESULTS:")
        for filename, result in results.items():
            status_icon = "✅" if result == "SUCCESS" else "❌"
            print(f"   {status_icon} {filename}: {result}")

        # Save test report
        report_file = self.output_dir / "test_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PIXEL ART PIPELINE TEST REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test files: {len(self.test_files)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(self.test_files)-successful}\n")
            f.write(
                f"Success rate: {successful/len(self.test_files)*100:.1f}%\n\n")

            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for filename, result in results.items():
                f.write(f"{result}: {filename}\n")

        print(f"\n📄 Test report saved: {report_file}")

        if successful == len(self.test_files):
            print("\n🎉 ALL TESTS PASSED! Ready for full batch processing.")
        elif successful > 0:
            print(
                f"\n⚠️  {successful} tests passed, {len(self.test_files)-successful} failed. Check logs for issues.")
        else:
            print("\n❌ ALL TESTS FAILED! Check ComfyUI setup and node installation.")


if __name__ == "__main__":
    tester = PixelArtPipelineTester()
    tester.run_test_pipeline()
