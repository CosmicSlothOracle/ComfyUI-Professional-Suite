#!/usr/bin/env python3
"""
🎨 ENHANCED GIF PROCESSOR - SDXL + LORA + 15-COLOR
==================================================
Combines SDXL + LoRA enhancement with 15-color palette conversion
"""

import json
import requests
import io
import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence


class EnhancedGIFProcessor:
    def __init__(self):
        self.server_url = "http://localhost:8188"
        self.client_id = str(uuid.uuid4())

        # 15-Color Palette from analysis
        self.palette_15_colors = [
            (8, 12, 16), (25, 15, 35), (85, 25, 35), (200, 100, 45),
            (235, 220, 180), (145, 220, 85), (45, 160, 95), (35, 85, 95),
            (25, 45, 85), (65, 115, 180), (85, 195, 215), (95, 105, 125),
            (85, 75, 95), (45, 55, 75), (12, 8, 8)
        ]

        print("🚀 Enhanced GIF Processor initialized")

    def apply_15_color_palette(self, image):
        """Apply 15-color palette to image"""
        img_array = np.array(image.convert('RGB'))
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3)
        palette_array = np.array(self.palette_15_colors)

        distances = np.sqrt(
            np.sum((pixels[:, None, :] - palette_array[None, :, :]) ** 2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        new_pixels = palette_array[closest_indices]
        new_img_array = new_pixels.reshape(original_shape).astype(np.uint8)

        return Image.fromarray(new_img_array)

    def enhance_with_sdxl_simple(self, input_image_path):
        """Simple SDXL enhancement for single image"""
        # Upload image first
        with open(input_image_path, 'rb') as f:
            files = {'image': ('input.png', f, 'image/png')}
            response = requests.post(
                f"{self.server_url}/upload/image", files=files)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name')
                print(f"📤 Image uploaded: {uploaded_name}")
            else:
                print("❌ Upload failed")
                return False

        # Create workflow
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sdxl.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pixel-art-xl-lora.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": "pixel art, 8bit style, retro game graphics, sharp pixels",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, photorealistic, smooth",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {"image": uploaded_name},
                "class_type": "LoadImage"
            },
            "6": {
                "inputs": {
                    "pixels": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEEncode"
            },
            "7": {
                "inputs": {
                    "seed": 42,
                    "steps": 8,
                    "cfg": 5.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.4,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "KSampler"
            },
            "8": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "enhanced_frame",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

        # Queue workflow
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            response = requests.post(f"{self.server_url}/prompt", json=payload)
            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get('prompt_id')
                print(f"✅ Enhancement queued: {prompt_id}")
                return True
            else:
                print(f"❌ Queue failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Enhancement failed: {e}")
            return False

    def process_gif_with_palette_only(self, input_path, output_path):
        """Fast processing with 15-color palette only"""
        print(f"🎬 Processing GIF with 15-color palette: {input_path}")

        try:
            with Image.open(input_path) as gif:
                frames = []
                durations = []
                has_transparency = False

                # Check transparency
                if hasattr(gif, 'info') and 'transparency' in gif.info:
                    has_transparency = True
                    transparency_color = gif.info['transparency']
                    print(f"🔍 Transparency detected: {transparency_color}")

                total_frames = getattr(gif, 'n_frames', 1)
                print(
                    f"📊 Processing {total_frames} frames with 15-color palette")

                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    # Preserve transparency
                    alpha_channel = None
                    if frame.mode in ('RGBA', 'LA') or 'transparency' in frame.info:
                        if frame.mode == 'P' and 'transparency' in frame.info:
                            frame = frame.convert('RGBA')
                        if frame.mode in ('RGBA', 'LA'):
                            alpha_channel = frame.split()[-1]

                    # Apply 15-color palette
                    rgb_frame = frame.convert('RGB')
                    palette_frame = self.apply_15_color_palette(rgb_frame)

                    # Restore transparency
                    if alpha_channel:
                        palette_frame = palette_frame.convert('RGBA')
                        palette_frame.putalpha(alpha_channel)
                    elif has_transparency and frame.mode == 'P':
                        palette_frame = palette_frame.convert('P')
                        palette_frame.info['transparency'] = transparency_color

                    frames.append(palette_frame)

                    # Preserve timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                    if (frame_num + 1) % 10 == 0:
                        print(
                            f"🖼️ Processed {frame_num + 1}/{total_frames} frames")

                # Save GIF
                if frames:
                    save_kwargs = {
                        'save_all': True,
                        'append_images': frames[1:],
                        'duration': durations,
                        'loop': 0,
                        'optimize': True
                    }

                    if has_transparency:
                        save_kwargs['transparency'] = 0
                        save_kwargs['disposal'] = 2

                    frames[0].save(output_path, **save_kwargs)

                    print(f"✅ Enhanced GIF saved: {output_path}")
                    print(f"📊 Total frames: {len(frames)}")
                    print(f"🎨 Applied 15-color palette to all frames")

                    return True

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return False

        return False


def test_enhanced_processing():
    """Test enhanced GIF processing"""
    processor = EnhancedGIFProcessor()

    # Test files
    test_files = [
        "input/chorizombi-umma.gif",
        "input/2a9bdd70dc936f3c482444e529694edf_fast_transparent_converted.gif"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🎯 Testing enhanced processing: {test_file}")

            output_file = f"output/ENHANCED_15COLOR_{Path(test_file).stem}.gif"

            success = processor.process_gif_with_palette_only(
                test_file, output_file)

            if success:
                print(f"🎉 Enhancement successful: {output_file}")

                # Optional: Test SDXL enhancement on first frame
                print("\n🔄 Testing SDXL enhancement on sample frame...")

                # Extract first frame for SDXL test
                with Image.open(test_file) as gif:
                    first_frame = gif.copy()
                    if first_frame.mode != 'RGB':
                        first_frame = first_frame.convert('RGB')

                    frame_path = "temp_frame_test.png"
                    first_frame.save(frame_path)

                    # Test SDXL enhancement
                    processor.enhance_with_sdxl_simple(frame_path)

                    # Clean up
                    Path(frame_path).unlink(missing_ok=True)

                return True
            else:
                print(f"❌ Enhancement failed")

    return False


def main():
    print("🎨 ENHANCED GIF PROCESSOR TEST")
    print("=" * 50)

    success = test_enhanced_processing()

    if success:
        print("\n✅ Enhanced processing test successful!")
        print("🎯 Ready for full enhanced pipeline!")
    else:
        print("\n❌ Test failed")


if __name__ == "__main__":
    main()
