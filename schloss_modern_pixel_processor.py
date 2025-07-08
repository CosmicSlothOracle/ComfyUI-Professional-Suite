#!/usr/bin/env python3
"""
🏰 SCHLOSS MODERN PIXEL ART PROCESSOR
===================================
Verwendet das EXAKTE Workflow von pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif
Konvertiert MP4 → Frames → Modern Pixel Art → GIF
"""

import json
import requests
import io
import time
import uuid
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageSequence


class SchlossModernPixelProcessor:
    def __init__(self):
        self.server_url = "http://localhost:8188"
        self.client_id = str(uuid.uuid4())

        # Original Modern Pixel Art Settings (exakt wie pixel_modern_*)
        self.modern_pixel_settings = {
            "palette": "GAMEBOY",
            "pixelize": "Image.quantize",
            "grid_pixelate_grid_scan_size": 2,
            "resize_w": 512,
            "resize_h": 512,
            "reduce_colors_before_palette_swap": True,
            "reduce_colors_max_colors": 64,
            "apply_pixeldetector_max_colors": True,
            "image_quantize_reduce_method": "MAXCOVERAGE",
            "opencv_settings": "",
            "opencv_kmeans_centers": "RANDOM_CENTERS",
            "opencv_kmeans_attempts": 10,
            "opencv_criteria_max_iterations": 10,
            "cleanup": "",
            "cleanup_colors": True,
            "cleanup_pixels_threshold": 0.02,
            "dither": "none"
        }

        print("🏰 Schloss Modern Pixel Processor initialized")
        print("🎯 Target: Schlossgifdance.mp4")
        print("🎨 Style: Modern Pixel Art (GAMEBOY Palette)")
        print("⚙️  Method: PixelArtDetectorConverter")

    def extract_frames_from_mp4(self, mp4_path, max_frames=None):
        """Extrahiere Frames aus MP4 Video"""
        try:
            cap = cv2.VideoCapture(mp4_path)

            # Video Info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            print(f"📹 MP4 Analysis:")
            print(f"   FPS: {fps}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Duration: {duration:.1f}s")

            if max_frames:
                total_frames = min(total_frames, max_frames)
                print(f"   Processing: {total_frames} frames (limited)")

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break

                # OpenCV BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"   Extracted: {frame_count}/{total_frames} frames")

            cap.release()

            # Frame Duration für GIF (aus FPS berechnet)
            frame_duration = int(1000 / fps)  # milliseconds

            print(f"✅ Extracted {len(frames)} frames")
            print(f"📊 Frame Duration: {frame_duration}ms")

            return frames, frame_duration

        except Exception as e:
            print(f"❌ MP4 extraction error: {e}")
            return None, None

    def upload_frame(self, frame, frame_num):
        """Upload Frame zu ComfyUI für Modern Pixel Art Processing"""
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        # Resize to 512x512 (Modern Pixel Art Standard)
        frame = frame.resize((512, 512), Image.LANCZOS)

        img_buffer = io.BytesIO()
        frame.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)

        files = {
            'image': (f'schloss_frame_{frame_num:04d}.png', img_buffer, 'image/png')}

        try:
            response = requests.post(
                f"{self.server_url}/upload/image", files=files)
            if response.status_code == 200:
                result = response.json()
                return result.get('name')
            return None
        except Exception as e:
            print(f"❌ Upload error frame {frame_num}: {e}")
            return None

    def create_modern_pixel_workflow(self, input_image_name, frame_num):
        """Erstelle EXAKTES Modern Pixel Art Workflow"""
        workflow = {
            "1": {
                "inputs": {
                    "image": input_image_name
                },
                "class_type": "LoadImage",
                "_meta": {
                    "title": "Load Input Image"
                }
            },
            "2": {
                "inputs": {
                    "images": ["1", 0],
                    **self.modern_pixel_settings
                },
                "class_type": "PixelArtDetectorConverter",
                "_meta": {
                    "title": "Modern Pixel Art Converter"
                }
            },
            "3": {
                "inputs": {
                    "images": ["2", 0],
                    "reduce_palette": False,
                    "reduce_palette_max_colors": 64
                },
                "class_type": "PixelArtDetectorToImage",
                "_meta": {
                    "title": "Convert to Image"
                }
            },
            "4": {
                "inputs": {
                    "images": ["3", 0],
                    "filename_prefix": f"schloss_modern_pixel_{frame_num:04d}",
                    "reduce_palette": False,
                    "reduce_palette_max_colors": 64,
                    "webp_mode": "lossless",
                    "compression": 90,
                    "save_jpg": False,
                    "save_exif": True,
                    "resize_w": 512,
                    "resize_h": 512
                },
                "class_type": "PixelArtDetectorSave",
                "_meta": {
                    "title": "Save Modern Pixel Art"
                }
            }
        }
        return workflow

    def queue_and_wait(self, workflow, frame_num, total_frames, timeout=300):
        """Queue Modern Pixel Art Processing"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        try:
            print(
                f"🎨 Processing Frame {frame_num}/{total_frames} with Modern Pixel Art...")
            response = requests.post(f"{self.server_url}/prompt", json=payload)

            if response.status_code != 200:
                print(f"❌ Queue failed for frame {frame_num}")
                return None

            result = response.json()
            prompt_id = result.get('prompt_id')

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
                            elapsed = time.time() - start_time
                            progress = frame_num / total_frames * 100
                            print(
                                f"   ✅ Frame {frame_num} complete in {elapsed:.1f}s! ({progress:.1f}% done)")
                            return prompt_id

                    time.sleep(2)

                except Exception:
                    time.sleep(2)

            print(f"⏰ Frame {frame_num} timeout")
            return None

        except Exception as e:
            print(f"❌ Frame {frame_num} error: {e}")
            return None

    def get_processed_frame(self, prompt_id):
        """Download Processed Modern Pixel Art Frame"""
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
                                processed_frame = Image.open(
                                    io.BytesIO(img_response.content))
                                return processed_frame
            return None

        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def process_schloss_mp4(self, input_path, output_path, max_frames=100):
        """Verarbeite Schlossgifdance.mp4 mit Modern Pixel Art Workflow"""
        print(f"🏰 SCHLOSS MODERN PIXEL ART PROCESSING")
        print("=" * 60)

        # Extract Frames from MP4
        frames, frame_duration = self.extract_frames_from_mp4(
            input_path, max_frames)
        if not frames:
            return False

        print(f"\n🎯 Processing {len(frames)} frames with Modern Pixel Art...")
        print("🚀 Starting Schloss processing...\n")

        try:
            processed_frames = []
            total_frames = len(frames)
            start_time = time.time()
            successful_frames = 0

            for frame_num, frame in enumerate(frames):
                print(f"\n🖼️ === FRAME {frame_num + 1}/{total_frames} ===")

                # Upload Frame
                uploaded_name = self.upload_frame(frame, frame_num)
                if not uploaded_name:
                    print(f"❌ Upload failed for frame {frame_num + 1}")
                    processed_frames.append(frame)
                    continue

                # Create Modern Pixel Art Workflow
                workflow = self.create_modern_pixel_workflow(
                    uploaded_name, frame_num)

                # Process with Modern Pixel Art
                prompt_id = self.queue_and_wait(
                    workflow, frame_num + 1, total_frames)
                if not prompt_id:
                    print(f"❌ Processing failed for frame {frame_num + 1}")
                    processed_frames.append(frame)
                    continue

                # Get Processed Frame
                processed_frame = self.get_processed_frame(prompt_id)
                if not processed_frame:
                    print(f"❌ Download failed for frame {frame_num + 1}")
                    processed_frames.append(frame)
                    continue

                processed_frames.append(processed_frame)
                successful_frames += 1

                # Progress Report
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

            # Save Modern Pixel Art GIF
            if processed_frames:
                print(f"\n💾 Saving Modern Pixel Art GIF...")

                processed_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=processed_frames[1:],
                    duration=frame_duration,
                    loop=0,
                    optimize=True
                )

                total_time = time.time() - start_time

                print(f"\n🎉 SCHLOSS MODERN PIXEL ART COMPLETE!")
                print("=" * 60)
                print(f"✅ Output GIF: {output_path}")
                print(f"📊 Final Statistics:")
                print(f"   Total Frames: {len(processed_frames)}")
                print(f"   Processed Frames: {successful_frames}")
                print(
                    f"   Success Rate: {successful_frames/len(processed_frames)*100:.1f}%")
                print(f"   Processing Time: {total_time/60:.1f} minutes")
                print(
                    f"   Average per Frame: {total_time/len(processed_frames):.1f} seconds")
                print(f"   Style: Modern Pixel Art (GAMEBOY Palette)")
                print(f"   Resolution: 512x512")
                print(f"   Frame Duration: {frame_duration}ms")

                return True

        except Exception as e:
            print(f"❌ Schloss processing error: {e}")
            return False

        return False


def main():
    """Process Schlossgifdance.mp4 with Modern Pixel Art workflow"""
    print("🏰 SCHLOSS MODERN PIXEL ART PROCESSOR")
    print("=" * 60)

    processor = SchlossModernPixelProcessor()

    # Target Files
    input_file = "input/Schlossgifdance.mp4"
    output_file = "output/SCHLOSS_MODERN_PIXEL_ART.gif"

    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return

    # Process mit max 100 Frames (für reasonable processing time)
    success = processor.process_schloss_mp4(
        input_file, output_file, max_frames=100)

    if success:
        print(f"\n🎉 SUCCESS! Modern Pixel Art Schloss ready: {output_file}")
        print(f"📸 Style matches: pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif")
    else:
        print(f"\n❌ Processing failed.")


if __name__ == "__main__":
    main()
