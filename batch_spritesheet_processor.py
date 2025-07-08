#!/usr/bin/env python3
"""
BATCH SPRITESHEET PROCESSOR
Verarbeitet alle Spritesheets im input Ordner mit dem Enhanced Spritesheet Processor
Features: Hintergrundentfernung, Weißabgleich, Hochskalierung, GIF-Erstellung
"""

import os
import sys
import json
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import List, Dict, Any

# ComfyUI API Setup
COMFYUI_BASE_URL = "http://127.0.0.1:8188"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/intelligent_sprites_batch")


class BatchSpritesheetProcessor:
    def __init__(self):
        self.session = None
        self.processed_count = 0
        self.failed_count = 0

    async def init_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    def create_enhanced_workflow(self, image_filename: str) -> Dict[str, Any]:
        """Create workflow for enhanced spritesheet processing"""
        return {
            "1": {
                "inputs": {
                    "image": image_filename,
                    "upload": "image"
                },
                "class_type": "LoadImage",
                "_meta": {"title": f"Load: {image_filename}"}
            },
            "2": {
                "inputs": {
                    "image": ["1", 0],
                    "adaptive_tolerance": True,
                    "tolerance_override": 35.0,
                    "aggressive_extraction": True,
                    "min_area_factor": 2500.0,
                    "edge_refinement": True,
                    "hsv_analysis": True,
                    "multi_zone_sampling": True,
                    "morphological_cleanup": True,
                    "smooth_edges": True
                },
                "class_type": "EnhancedSpritesheetProcessor",
                "_meta": {"title": "🎮 Enhanced Spritesheet Processor"}
            },
            "3": {
                "inputs": {
                    "images": ["2", 0],
                    "filename_prefix": f"sprite_{Path(image_filename).stem}_frame"
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Enhanced Frames"}
            },
            "4": {
                "inputs": {
                    "text": ["2", 1]
                },
                "class_type": "ShowText",
                "_meta": {"title": "Processing Info"}
            },
            "5": {
                "inputs": {
                    "text": ["2", 2]
                },
                "class_type": "ShowText",
                "_meta": {"title": "Frame Statistics"}
            }
        }

    async def check_comfyui_status(self):
        """Check if ComfyUI is running"""
        try:
            async with self.session.get(f"{COMFYUI_BASE_URL}/system_stats") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def upload_image(self, image_path: Path) -> bool:
        """Upload image to ComfyUI"""
        try:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('image', f, filename=image_path.name)

                async with self.session.post(f"{COMFYUI_BASE_URL}/upload/image", data=data) as resp:
                    if resp.status == 200:
                        print(f"✅ Uploaded: {image_path.name}")
                        return True
                    else:
                        print(
                            f"❌ Upload failed for {image_path.name}: {resp.status}")
                        return False
        except Exception as e:
            print(f"❌ Upload error for {image_path.name}: {e}")
            return False

    async def queue_workflow(self, workflow: Dict[str, Any]) -> str:
        """Queue workflow for processing"""
        try:
            payload = {"prompt": workflow}
            async with self.session.post(f"{COMFYUI_BASE_URL}/prompt", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("prompt_id", "")
                else:
                    print(f"❌ Workflow queue failed: {resp.status}")
                    return ""
        except Exception as e:
            print(f"❌ Workflow queue error: {e}")
            return ""

    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """Wait for workflow completion"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            if status.get("completed", False):
                                print("✅ Processing completed!")
                                return True
                            if status.get("status_str") == "error":
                                print("❌ Processing failed!")
                                return False

                await asyncio.sleep(2)
            except Exception as e:
                print(f"⚠️ Status check error: {e}")
                await asyncio.sleep(5)

        print("⏰ Timeout reached!")
        return False

    def get_spritesheet_files(self) -> List[Path]:
        """Get all spritesheet files from input directory"""
        sprite_files = []

        # Main input directory
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            sprite_files.extend(INPUT_DIR.glob(ext))

        # Sprite sheets subdirectory
        sprite_sheets_dir = INPUT_DIR / "sprite_sheets"
        if sprite_sheets_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                sprite_files.extend(sprite_sheets_dir.glob(ext))

        # Filter out small files (likely thumbnails)
        valid_files = []
        for file in sprite_files:
            try:
                if file.stat().st_size > 100_000:  # > 100KB
                    valid_files.append(file)
                    print(
                        f"📋 Found spritesheet: {file.name} ({file.stat().st_size // 1024}KB)")
            except Exception:
                continue

        return valid_files

    async def process_single_spritesheet(self, image_path: Path) -> bool:
        """Process a single spritesheet"""
        print(f"\n🎮 Processing: {image_path.name}")

        # Upload image
        if not await self.upload_image(image_path):
            return False

        # Create and queue workflow
        workflow = self.create_enhanced_workflow(image_path.name)
        prompt_id = await self.queue_workflow(workflow)

        if not prompt_id:
            return False

        print(f"📋 Queued workflow with ID: {prompt_id}")

        # Wait for completion
        success = await self.wait_for_completion(prompt_id)

        if success:
            self.processed_count += 1
            print(f"✅ Successfully processed: {image_path.name}")
        else:
            self.failed_count += 1
            print(f"❌ Failed to process: {image_path.name}")

        return success

    async def process_all_spritesheets(self):
        """Process all spritesheets in batch"""
        print("🚀 Starting Batch Spritesheet Processing...")
        print(f"📂 Input directory: {INPUT_DIR.absolute()}")
        print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize session
        await self.init_session()

        try:
            # Check ComfyUI status
            if not await self.check_comfyui_status():
                print("❌ ComfyUI is not running! Please start ComfyUI first.")
                return

            print("✅ ComfyUI is running")

            # Get all spritesheet files
            sprite_files = self.get_spritesheet_files()

            if not sprite_files:
                print("❌ No spritesheet files found in input directory!")
                return

            print(f"📊 Found {len(sprite_files)} spritesheet files to process")

            # Process each file
            for i, sprite_file in enumerate(sprite_files, 1):
                print(f"\n[{i}/{len(sprite_files)}] " + "="*50)
                await self.process_single_spritesheet(sprite_file)

                # Small delay between files
                await asyncio.sleep(1)

            # Final statistics
            print("\n" + "="*70)
            print("🎉 BATCH PROCESSING COMPLETE!")
            print(f"✅ Successfully processed: {self.processed_count}")
            print(f"❌ Failed: {self.failed_count}")
            print(f"📊 Total files: {len(sprite_files)}")
            print(
                f"💯 Success rate: {(self.processed_count / len(sprite_files) * 100):.1f}%")

        finally:
            await self.close_session()


async def main():
    """Main execution function"""
    processor = BatchSpritesheetProcessor()
    await processor.process_all_spritesheets()

if __name__ == "__main__":
    print("🎮 BATCH SPRITESHEET PROCESSOR")
    print("=" * 70)
    print("Features:")
    print("• Automatic background removal")
    print("• Enhanced edge detection")
    print("• HSV color analysis")
    print("• Morphological cleanup")
    print("• Multi-zone sampling")
    print("• Aggressive frame extraction")
    print("=" * 70)

    asyncio.run(main())
