#!/usr/bin/env python3
"""
🎨 ENHANCED GIF PROCESSOR - SDXL + LORA + 15-COLOR PALETTE
==========================================================
Pipeline that combines:
1. SDXL + Pixel-Art LoRA Enhancement for quality
2. 15-Color Palette Conversion for authentic retro look
3. Transparency preservation for original animation feel
"""

import json
import requests
import io
import time
import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence


class EnhancedGIFProcessor:
    def __init__(self, server_url="http://localhost:8188"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())

        # SDXL Models
        self.sdxl_checkpoint = "sdxl.safetensors"
        self.pixelart_lora = "pixel-art-xl-lora.safetensors"

        # 15-Color Palette (from your analysis)
        self.palette_15_colors = [
            (8, 12, 16),      # Deep black with blue tint
            (25, 15, 35),     # Very dark purple
            (85, 25, 35),     # Dark red/wine
            (200, 100, 45),   # Rich orange
            (235, 220, 180),  # Pale yellow/beige
            (145, 220, 85),   # Light apple green
            (45, 160, 95),    # Richer emerald green
            (35, 85, 95),     # Dark green-blue
            (25, 45, 85),     # Navy blue
            (65, 115, 180),   # Medium blue
            (85, 195, 215),   # Cyan/turquoise
            (95, 105, 125),   # Gray-blue
            (85, 75, 95),     # Gray-purple
            (45, 55, 75),     # Graphite blue
            (12, 8, 8)        # Almost black variant
        ]

        # Performance Settings
        self.enhance_every_nth_frame = 3  # Enhance every 3rd frame for performance

        print("🚀 Enhanced GIF Processor initialized")
        self.test_connection()

    def test_connection(self):
        """Test ComfyUI server connection"""
        try:
            response = requests.get(f"{self.server_url}/system_stats")
            if response.status_code == 200:
                print("✅ ComfyUI Server connected!")
                return True
            else:
                print(f"❌ Server error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def apply_15_color_palette(self, image):
        """Apply the 15-color palette to an image"""
        # Convert to RGB array
        img_array = np.array(image.convert('RGB'))
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3)

        # Find closest palette color for each pixel
        palette_array = np.array(self.palette_15_colors)
        distances = np.sqrt(
            np.sum((pixels[:, None, :] - palette_array[None, :, :]) ** 2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        new_pixels = palette_array[closest_indices]
        new_img_array = new_pixels.reshape(original_shape).astype(np.uint8)

        return Image.fromarray(new_img_array)

    def upload_image_to_comfyui(self, image):
        """Upload image to ComfyUI for SDXL processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        files = {'image': ('input.png', img_buffer, 'image/png')}

        try:
            response = requests.post(
                f"{self.server_url}/upload/image", files=files)
            if response.status_code == 200:
                result = response.json()
                return result.get('name')
            return None
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None

    def create_sdxl_workflow(self, input_image_name):
        """Create SDXL + LoRA enhancement workflow"""
        workflow = {
            "1": {
                "inputs": {"ckpt_name": self.sdxl_checkpoint},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": self.pixelart_lora,
                    "strength_model": 0.9,
                    "strength_clip": 0.9,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": "pixel art, 8bit style, retro game graphics, sharp pixels, limited color palette, clean pixel work, crisp edges",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, photorealistic, smooth gradients, anti-aliasing, high resolution, soft edges",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {"image": input_image_name},
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
                    "seed": int(time.time()) % 1000000,
                    "steps": 8,
                    "cfg": 4.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.35,
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
                    "filename_prefix": f"enhanced_{int(time.time())}",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow

    def queue_and_wait_enhancement(self, workflow, timeout=120):
        """Queue workflow and wait for completion"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            response = requests.post(f"{self.server_url}/prompt", json=payload)
            if response.status_code != 200:
                return None

            result = response.json()
            prompt_id = result.get('prompt_id')

            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = requests.get(f"{self.server_url}/queue")
                if response.status_code == 200:
                    queue_data = response.json()
                    running = queue_data.get('queue_running', [])
                    pending = queue_data.get('queue_pending', [])

                    is_running = any(item[1] == prompt_id for item in running)
                    is_pending = any(item[1] == prompt_id for item in pending)

                    if not is_running and not is_pending:
                        return prompt_id

                time.sleep(2)

            return None

        except Exception as e:
            print(f"❌ Enhancement error: {e}")
            return None

    def get_enhanced_image(self, prompt_id):
        """Get enhanced image from ComfyUI server"""
        try:
            response = requests.get(f"{self.server_url}/history/{prompt_id}")
            if response.status_code != 200:
                return None

            history = response.json()
            if prompt_id not in history:
                return None

            outputs = history[prompt_id].get('outputs', {})

            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    for img_info in node_output['images']:
                        filename = img_info.get('filename')
                        if filename:
                            img_response = requests.get(
                                f"{self.server_url}/view",
                                params={'filename': filename}
                            )
                            if img_response.status_code == 200:
                                return Image.open(io.BytesIO(img_response.content))

            return None

        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def enhance_frame_with_sdxl(self, frame):
        """Enhance single frame with SDXL + LoRA"""
        original_size = frame.size

        # Resize for SDXL if too small
        if max(frame.size) < 512:
            scale = 512 / max(frame.size)
            new_size = (
                int(frame.size[0] * scale),
                int(frame.size[1] * scale)
            )
            frame = frame.resize(new_size, Image.NEAREST)

        # Upload and process
        uploaded_name = self.upload_image_to_comfyui(frame)
        if not uploaded_name:
            return frame

        workflow = self.create_sdxl_workflow(uploaded_name)
        prompt_id = self.queue_and_wait_enhancement(workflow)

        if prompt_id:
            enhanced = self.get_enhanced_image(prompt_id)
            if enhanced:
                # Resize back to original
                if enhanced.size != original_size:
                    enhanced = enhanced.resize(original_size, Image.NEAREST)
                return enhanced

        return frame

    def should_enhance_frame(self, frame_num, total_frames):
        """Determine if frame should be enhanced"""
        # Enhance key frames: first, last, and every nth frame
        if frame_num == 0 or frame_num == total_frames - 1:
            return True
        if frame_num % self.enhance_every_nth_frame == 0:
            return True
        return False

    def process_enhanced_gif(self, input_path, output_path):
        """Process GIF with enhanced pipeline"""
        print(f"🎬 Enhanced GIF Processing: {input_path}")

        try:
            with Image.open(input_path) as gif:
                frames = []
                durations = []
                has_transparency = False

                # Check for transparency
                if hasattr(gif, 'info') and 'transparency' in gif.info:
                    has_transparency = True
                    transparency_color = gif.info['transparency']
                    print(
                        f"🔍 Transparency detected: Index {transparency_color}")

                total_frames = getattr(gif, 'n_frames', 1)
                print(f"📊 Processing {total_frames} frames")
                print(
                    f"🎯 Enhancement Strategy: Every {self.enhance_every_nth_frame}rd frame")

                enhanced_frames = {}

                # First pass: Enhance selected frames
                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    if self.should_enhance_frame(frame_num, total_frames):
                        print(
                            f"🔄 Enhancing Frame {frame_num + 1}/{total_frames} with SDXL...")

                        # Preserve transparency info
                        alpha_channel = None
                        if frame.mode in ('RGBA', 'LA') or 'transparency' in frame.info:
                            if frame.mode == 'P' and 'transparency' in frame.info:
                                frame = frame.convert('RGBA')
                            if frame.mode in ('RGBA', 'LA'):
                                alpha_channel = frame.split()[-1]

                        # Enhance frame
                        rgb_frame = frame.convert('RGB')
                        enhanced_frame = self.enhance_frame_with_sdxl(
                            rgb_frame)

                        # Apply 15-color palette
                        palette_frame = self.apply_15_color_palette(
                            enhanced_frame)

                        # Restore transparency
                        if alpha_channel:
                            palette_frame = palette_frame.convert('RGBA')
                            palette_frame.putalpha(alpha_channel)

                        enhanced_frames[frame_num] = palette_frame
                        print(
                            f"✅ Frame {frame_num + 1} enhanced and palette-converted")

                # Second pass: Process all frames
                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    if frame_num in enhanced_frames:
                        final_frame = enhanced_frames[frame_num]
                    else:
                        # Apply 15-color palette to original
                        alpha_channel = None
                        if frame.mode in ('RGBA', 'LA') or 'transparency' in frame.info:
                            if frame.mode == 'P' and 'transparency' in frame.info:
                                frame = frame.convert('RGBA')
                            if frame.mode in ('RGBA', 'LA'):
                                alpha_channel = frame.split()[-1]

                        rgb_frame = frame.convert('RGB')
                        palette_frame = self.apply_15_color_palette(rgb_frame)

                        if alpha_channel:
                            palette_frame = palette_frame.convert('RGBA')
                            palette_frame.putalpha(alpha_channel)

                        final_frame = palette_frame

                    frames.append(final_frame)

                    # Preserve timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                    print(f"🖼️ Frame {frame_num + 1}/{total_frames} processed")

                # Save enhanced GIF
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

                    print(f"\n✅ Enhanced GIF saved: {output_path}")
                    print(f"📊 Statistics:")
                    print(f"   Total Frames: {len(frames)}")
                    print(f"   Enhanced Frames: {len(enhanced_frames)}")
                    print(
                        f"   Enhancement Ratio: {len(enhanced_frames)/len(frames)*100:.1f}%")
                    print(
                        f"   Transparency: {'✅' if has_transparency else '❌'}")

                    return True

        except Exception as e:
            print(f"❌ Enhanced processing error: {e}")
            return False

        return False


def main():
    """Demo Enhanced GIF Processing"""
    print("🎨 ENHANCED GIF PROCESSOR - SDXL + LORA + 15-COLOR")
    print("=" * 60)

    processor = EnhancedGIFProcessor()

    # Test files
    test_files = [
        "input/chorizombi-umma.gif",
        "input/2a9bdd70dc936f3c482444e529694edf_fast_transparent_converted.gif",
        "input/7ee80664e6f86ac416750497557bf6fc_fast_transparent_converted.gif"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🎯 Processing: {test_file}")

            output_file = f"output/ENHANCED_SDXL_15COLOR_{Path(test_file).stem}.gif"

            success = processor.process_enhanced_gif(test_file, output_file)

            if success:
                print(f"🎉 Enhancement successful: {output_file}")
            else:
                print(f"❌ Enhancement failed")

            break  # Test only first available file

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
