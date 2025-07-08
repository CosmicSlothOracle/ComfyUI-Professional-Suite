#!/usr/bin/env python3
"""
🎨 CHORIZOMBI FULL SDXL PROCESSOR
================================
Speziell für: chorizombi-umma.gif (80 Frames)
JEDES Frame wird mit SDXL + LoRA enhanced!
"""

import json
import requests
import io
import time
import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence


class ChorizombiSDXLProcessor:
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

        print("🎮 Chorizombi SDXL Processor initialized")
        print("🎯 Target: chorizombi-umma.gif (80 frames)")
        print("🔥 Mode: EVERY FRAME enhanced with SDXL + LoRA")

    def analyze_gif(self, gif_path):
        """Analysiere das Chorizombi GIF"""
        try:
            with Image.open(gif_path) as gif:
                total_frames = getattr(gif, 'n_frames', 1)
                size = gif.size

                # Realistischer Zeitschätzung für 80 Frames
                # 4 Minuten pro Frame (optimistisch)
                estimated_time = total_frames * 4

                print(f"📊 Chorizombi GIF Analysis:")
                print(f"   Frames: {total_frames}")
                print(f"   Size: {size}")
                print(
                    f"   Estimated Time: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
                print(f"   Expected Quality: MAXIMUM")

                return {
                    'frames': total_frames,
                    'size': size,
                    'estimated_minutes': estimated_time
                }

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None

    def apply_15_color_palette(self, image):
        """Apply 15-color palette optimized for character animation"""
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
        """Upload Frame zu ComfyUI mit Optimierung"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        # Chorizombi ist bereits gute Größe, minimal anpassen
        target_size = 512
        if max(frame.size) != target_size:
            scale = target_size / max(frame.size)
            new_size = (
                int(frame.size[0] * scale),
                int(frame.size[1] * scale)
            )
            # Bessere Qualität für Character
            frame = frame.resize(new_size, Image.LANCZOS)

        img_buffer = io.BytesIO()
        frame.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)

        files = {'image': ('chorizombi_frame.png', img_buffer, 'image/png')}

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

    def create_character_workflow(self, input_image_name, frame_num):
        """Erstelle Character-optimiertes SDXL + LoRA Workflow"""
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sdxl.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pixel-art-xl-lora.safetensors",
                    "strength_model": 1.0,  # Maximale LoRA für Character
                    "strength_clip": 1.0,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": ("masterpiece pixel art character, 8bit retro game sprite, sharp crisp pixels, "
                             "limited color palette, clean character design, detailed pixel animation, "
                             "vintage arcade character, perfect pixel alignment, classic sprite art, "
                             "retro gaming character, smooth animation frame"),
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": ("blurry, photorealistic, smooth gradients, anti-aliasing, "
                             "high resolution, soft edges, modern 3D graphics, realistic textures, "
                             "complex shading, detailed background, noise, artifacts, "
                             "oversaturated, too many colors"),
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
                    # Consistent but varied seeds
                    "seed": (1000 + frame_num) % 1000000,
                    "steps": 12,  # Mehr Steps für Character-Detail
                    "cfg": 4.0,   # Optimaler CFG für Character
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 0.45,  # Mittlerer Denoise für Animation-Konsistenz
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
                    "filename_prefix": f"chorizombi_sdxl_{frame_num:04d}",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow

    def queue_and_wait_with_progress(self, workflow, frame_num, total_frames, timeout=600):
        """Queue mit detailliertem Progress für große Animation"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            print(
                f"🔄 Processing Frame {frame_num}/{total_frames} with SDXL + LoRA...")
            response = requests.post(f"{self.server_url}/prompt", json=payload)

            if response.status_code != 200:
                print(f"❌ Queue failed for frame {frame_num}")
                return None

            result = response.json()
            prompt_id = result.get('prompt_id')

            # Warte mit Progress-Updates
            start_time = time.time()
            last_update = start_time

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

                        # Progress Update alle 20 Sekunden
                        if time.time() - last_update > 20:
                            elapsed = time.time() - start_time
                            print(
                                f"   ⏳ Frame {frame_num}: {elapsed:.0f}s elapsed...")
                            last_update = time.time()

                        if not is_running and not is_pending:
                            elapsed = time.time() - start_time
                            progress = frame_num / total_frames * 100
                            print(
                                f"   ✅ Frame {frame_num} complete in {elapsed:.0f}s! ({progress:.1f}% done)")
                            return prompt_id

                    time.sleep(3)  # Check every 3 seconds for responsiveness

                except Exception:
                    time.sleep(3)

            print(f"⏰ Frame {frame_num} timeout")
            return None

        except Exception as e:
            print(f"❌ Frame {frame_num} error: {e}")
            return None

    def get_enhanced_frame(self, prompt_id, original_size):
        """Download Enhanced Frame mit Error Handling"""
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

                                # Resize zurück zu Original mit hoher Qualität
                                if enhanced.size != original_size:
                                    enhanced = enhanced.resize(
                                        original_size, Image.LANCZOS)

                                return enhanced
            return None

        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def process_chorizombi_gif(self, input_path, output_path):
        """Verarbeite Chorizombi GIF mit Full SDXL Enhancement"""
        print(f"🎮 CHORIZOMBI FULL SDXL PROCESSING")
        print("=" * 60)

        # Analyze first
        analysis = self.analyze_gif(input_path)
        if not analysis:
            return False

        print(
            f"\n⚠️  This will take approximately {analysis['estimated_minutes']} minutes!")
        print(
            f"🎯 Processing {analysis['frames']} frames with maximum quality...")
        print("🚀 Starting Chorizombi SDXL processing...\n")

        try:
            with Image.open(input_path) as gif:
                enhanced_frames = []
                durations = []

                total_frames = analysis['frames']
                start_time = time.time()
                successful_frames = 0

                for frame_num, frame in enumerate(ImageSequence.Iterator(gif)):
                    print(f"\n🖼️ === FRAME {frame_num + 1}/{total_frames} ===")

                    original_size = frame.size
                    rgb_frame = frame.convert('RGB')

                    # Upload Frame
                    uploaded_name = self.upload_frame(rgb_frame)
                    if not uploaded_name:
                        print(f"❌ Upload failed for frame {frame_num + 1}")
                        enhanced_frames.append(frame)
                        continue

                    # Create Workflow
                    workflow = self.create_character_workflow(
                        uploaded_name, frame_num)

                    # Process with SDXL
                    prompt_id = self.queue_and_wait_with_progress(
                        workflow, frame_num + 1, total_frames)
                    if not prompt_id:
                        print(f"❌ SDXL failed for frame {frame_num + 1}")
                        enhanced_frames.append(frame)
                        continue

                    # Get Enhanced Frame
                    enhanced_frame = self.get_enhanced_frame(
                        prompt_id, original_size)
                    if not enhanced_frame:
                        print(f"❌ Download failed for frame {frame_num + 1}")
                        enhanced_frames.append(frame)
                        continue

                    # Apply 15-Color Palette
                    print(
                        f"🎨 Applying 15-color palette to frame {frame_num + 1}...")
                    palette_frame = self.apply_15_color_palette(enhanced_frame)
                    enhanced_frames.append(palette_frame)
                    successful_frames += 1

                    # Preserve timing
                    duration = getattr(frame, 'info', {}).get('duration', 100)
                    durations.append(duration)

                    # Detailed Progress Report
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (frame_num + 1)
                    eta = (total_frames - frame_num - 1) * avg_time
                    success_rate = successful_frames / (frame_num + 1) * 100

                    print(f"📊 Progress Report:")
                    print(
                        f"   ✅ Frames Complete: {frame_num + 1}/{total_frames} ({(frame_num + 1)/total_frames*100:.1f}%)")
                    print(f"   🎯 Success Rate: {success_rate:.1f}%")
                    print(
                        f"   ⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
                    print(f"   📈 Avg per Frame: {avg_time:.1f}s")

                # Save Enhanced Chorizombi GIF
                if enhanced_frames:
                    print(f"\n💾 Saving enhanced Chorizombi GIF...")

                    enhanced_frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=enhanced_frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )

                    total_time = time.time() - start_time

                    print(f"\n🎉 CHORIZOMBI SDXL PROCESSING COMPLETE!")
                    print("=" * 60)
                    print(f"✅ Enhanced GIF: {output_path}")
                    print(f"📊 Final Statistics:")
                    print(f"   Total Frames: {len(enhanced_frames)}")
                    print(f"   Enhanced Frames: {successful_frames}")
                    print(
                        f"   Success Rate: {successful_frames/len(enhanced_frames)*100:.1f}%")
                    print(f"   Processing Time: {total_time/60:.1f} minutes")
                    print(
                        f"   Average per Frame: {total_time/len(enhanced_frames):.1f} seconds")
                    print(f"   Quality Level: MAXIMUM (SDXL + LoRA + 15-Color)")

                    return True

        except Exception as e:
            print(f"❌ Chorizombi processing error: {e}")
            return False

        return False


def main():
    """Process chorizombi-umma.gif with full SDXL enhancement"""
    print("🎮 CHORIZOMBI FULL SDXL + LORA PROCESSOR")
    print("=" * 60)

    processor = ChorizombiSDXLProcessor()

    # Target GIF
    input_file = "input/chorizombi-umma.gif"
    output_file = "output/CHORIZOMBI_FULL_SDXL_ENHANCED.gif"

    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return

    success = processor.process_chorizombi_gif(input_file, output_file)

    if success:
        print(f"\n🎉 SUCCESS! Enhanced Chorizombi ready: {output_file}")
    else:
        print(f"\n❌ Processing failed.")


if __name__ == "__main__":
    main()
