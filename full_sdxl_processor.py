#!/usr/bin/env python3
"""
🎨 FULL SDXL + LORA PROCESSOR
============================
Verarbeitet JEDES einzelne Frame mit SDXL + LoRA für maximale Qualität
Speziell für: c64a376922e4455d996537e81daf6512..GIF
"""

import json
import requests
import io
import time
import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence


class FullSDXLProcessor:
    def __init__(self):
        self.server_url = "http://localhost:8188"
        self.client_id = str(uuid.uuid4())

        # 15-Color Palette
        self.palette_15_colors = [
            (8, 12, 16), (25, 15, 35), (85, 25, 35), (200, 100, 45),
            (235, 220, 180), (145, 220, 85), (45, 160, 95), (35, 85, 95),
            (25, 45, 85), (65, 115, 180), (85, 195, 215), (95, 105, 125),
            (85, 75, 95), (45, 55, 75), (12, 8, 8)
        ]

        print("🚀 Full SDXL + LoRA Processor initialized")
        print("🎯 Mode: EVERY FRAME enhanced with SDXL + LoRA")

    def analyze_gif(self, gif_path):
        """Analysiere GIF für Processing-Plan"""
        try:
            with Image.open(gif_path) as gif:
                total_frames = getattr(gif, 'n_frames', 1)
                size = gif.size

                # Estimate processing time (5-8 Minuten pro Frame auf CPU)
                estimated_time = total_frames * 6  # Minuten

                print(f"📊 GIF Analysis:")
                print(f"   Frames: {total_frames}")
                print(f"   Size: {size}")
                print(
                    f"   Estimated Time: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")

                return {
                    'frames': total_frames,
                    'size': size,
                    'estimated_minutes': estimated_time
                }

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None

    def apply_15_color_palette(self, image):
        """Apply 15-color palette after SDXL enhancement"""
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

    def upload_frame(self, frame):
        """Upload Frame zu ComfyUI"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        # Resize für SDXL falls zu klein
        if max(frame.size) < 512:
            scale = 512 / max(frame.size)
            new_size = (
                int(frame.size[0] * scale),
                int(frame.size[1] * scale)
            )
            frame = frame.resize(new_size, Image.NEAREST)

        img_buffer = io.BytesIO()
        frame.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        files = {'image': ('frame.png', img_buffer, 'image/png')}

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

    def create_workflow(self, input_image_name, frame_num):
        """Erstelle SDXL + LoRA Workflow"""
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sdxl.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pixel-art-xl-lora.safetensors",
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": "masterpiece pixel art, 8bit retro game style, sharp crisp pixels, limited color palette, clean pixel work, vintage arcade aesthetic",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, photorealistic, smooth gradients, anti-aliasing, high resolution, soft edges, modern graphics",
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
                    "seed": (42 + frame_num) % 1000000,
                    "steps": 10,
                    "cfg": 4.5,
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
                    "filename_prefix": f"full_sdxl_frame_{frame_num:04d}",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow

    def queue_and_wait(self, workflow, frame_num, timeout=600):
        """Queue und warte auf Completion"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            print(f"🔄 Processing Frame {frame_num} with SDXL...")
            response = requests.post(f"{self.server_url}/prompt", json=payload)

            if response.status_code != 200:
                return None

            result = response.json()
            prompt_id = result.get('prompt_id')

            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{self.server_url}/queue")
                    if response.status_code == 200:
                        queue_data = response.json()
                        running = queue_data.get('queue_running', [])
                        pending = queue_data.get('queue_pending', [])

                        is_running = any(
                            item[1] == prompt_id for item in running)
                        is_pending = any(
                            item[1] == prompt_id for item in pending)

                        if not is_running and not is_pending:
                            return prompt_id

                    time.sleep(5)

                except Exception:
                    time.sleep(5)

            return None

        except Exception as e:
            print(f"❌ Enhancement error: {e}")
            return None

    def get_enhanced_frame(self, prompt_id, original_size):
        """Download Enhanced Frame"""
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
                                enhanced = Image.open(
                                    io.BytesIO(img_response.content))

                                # Resize zurück zu Original
                                if enhanced.size != original_size:
                                    enhanced = enhanced.resize(
                                        original_size, Image.NEAREST)

                                return enhanced
            return None

        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def process_gif(self, input_path, output_path):
        """Verarbeite GIF mit Full SDXL Enhancement"""
        print(f"🎬 FULL SDXL PROCESSING: {input_path}")

        # Analyze first
        analysis = self.analyze_gif(input_path)
        if not analysis:
            return False

        print(
            f"\n⚠️  This will take approximately {analysis['estimated_minutes']} minutes!")
        print("🚀 Starting FULL SDXL processing...")

        try:
            with Image.open(input_path) as gif:
                enhanced_frames = []
                durations = []

                total_frames = analysis['frames']
                start_time = time.time()

                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    print(f"\n🖼️ Frame {frame_num + 1}/{total_frames}")

                    original_size = frame.size
                    rgb_frame = frame.convert('RGB')

                    # Upload Frame
                    uploaded_name = self.upload_frame(rgb_frame)
                    if not uploaded_name:
                        enhanced_frames.append(frame)
                        continue

                    # Create Workflow
                    workflow = self.create_workflow(uploaded_name, frame_num)

                    # Process with SDXL
                    prompt_id = self.queue_and_wait(workflow, frame_num + 1)
                    if not prompt_id:
                        enhanced_frames.append(frame)
                        continue

                    # Get Enhanced Frame
                    enhanced_frame = self.get_enhanced_frame(
                        prompt_id, original_size)
                    if not enhanced_frame:
                        enhanced_frames.append(frame)
                        continue

                    # Apply 15-Color Palette
                    palette_frame = self.apply_15_color_palette(enhanced_frame)
                    enhanced_frames.append(palette_frame)

                    # Preserve timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                    # Progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (frame_num + 1)
                    eta = (total_frames - frame_num - 1) * avg_time

                    print(
                        f"✅ Frame {frame_num + 1} complete! ETA: {eta/60:.1f}min")

                # Save Enhanced GIF
                if enhanced_frames:
                    enhanced_frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=enhanced_frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )

                    total_time = time.time() - start_time
                    print(f"\n🎉 FULL SDXL PROCESSING COMPLETE!")
                    print(f"✅ Enhanced GIF: {output_path}")
                    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
                    return True

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return False

        return False


def main():
    """Process the specific GIF with full SDXL enhancement"""
    print("🎨 FULL SDXL + LORA PROCESSOR")
    print("=" * 50)

    processor = FullSDXLProcessor()

    # Target GIF
    input_file = "input/c64a376922e4455d996537e81daf6512..GIF"
    output_file = "output/FULL_SDXL_c64a376922e4455d996537e81daf6512.gif"

    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return

    success = processor.process_gif(input_file, output_file)

    if success:
        print(f"\n🎉 SUCCESS! Check: {output_file}")
    else:
        print(f"\n❌ Processing failed.")


if __name__ == "__main__":
    main()
