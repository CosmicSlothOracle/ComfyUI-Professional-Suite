#!/usr/bin/env python3
"""
🏰 SCHLOSS DIRECT PIXEL ART PROCESSOR
===================================
Direkte Python-Implementation des Modern Pixel Art Workflows
OHNE ComfyUI Custom Nodes - funktioniert standalone!
Nachbau von pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time


class SchlossDirectPixelProcessor:
    def __init__(self):
        # GAMEBOY Palette (wie im Original Modern Pixel Art)
        self.gameboy_palette = [
            (15, 56, 15),      # Dunkelgrün
            (48, 98, 48),      # Mittelgrün
            (139, 172, 15),    # Hellgrün
            (155, 188, 15)     # Sehr hellgrün
        ]

        print("🏰 Schloss Direct Pixel Processor initialized")
        print("🎯 Target: Schlossgifdance.mp4")
        print("🎨 Style: Modern Pixel Art (GAMEBOY Palette)")
        print("⚙️  Method: Direct Python Implementation")

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

    def reduce_colors_kmeans(self, image, max_colors=64):
        """Reduziere Farben mit K-Means (wie im Original Workflow)"""
        img_array = np.array(image)
        original_shape = img_array.shape

        # Reshape zu 2D array
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        # K-Means Clustering
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            max_colors,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Ersetze Pixel mit Cluster-Zentren
        centers = np.uint8(centers)
        reduced_pixels = centers[labels.flatten()]
        reduced_img = reduced_pixels.reshape(original_shape)

        return Image.fromarray(reduced_img)

    def apply_gameboy_palette(self, image):
        """Wende GAMEBOY Palette an (exakt wie Modern Pixel Art)"""
        img_array = np.array(image.convert('RGB'))
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3)
        palette_array = np.array(self.gameboy_palette)

        # Finde nächste Farbe für jeden Pixel
        distances = np.sqrt(
            np.sum((pixels[:, None, :] - palette_array[None, :, :]) ** 2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        new_pixels = palette_array[closest_indices]
        new_img_array = new_pixels.reshape(original_shape).astype(np.uint8)

        return Image.fromarray(new_img_array)

    def pixelize_with_quantize(self, image, grid_size=2):
        """Pixelize mit Image.quantize (wie im Original)"""
        # Resize runter für Pixel-Effekt
        small_size = (image.size[0] // grid_size, image.size[1] // grid_size)
        small_img = image.resize(small_size, Image.NEAREST)

        # Quantize (Farbreduktion)
        quantized = small_img.quantize(colors=64, method=Image.MAXCOVERAGE)

        # Zurück zu Original-Größe mit NEAREST (harte Pixel)
        pixelized = quantized.resize(image.size, Image.NEAREST)

        # Konvertiere zurück zu RGB
        return pixelized.convert('RGB')

    def cleanup_pixels(self, image, threshold=0.02):
        """Cleanup isolated pixels (wie im Original)"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Simple noise reduction
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

        return Image.fromarray(cleaned)

    def apply_modern_pixel_art(self, frame):
        """Komplette Modern Pixel Art Pipeline"""
        # 1. Resize to 512x512 (Standard)
        frame = frame.resize((512, 512), Image.LANCZOS)

        # 2. Reduce colors before palette swap
        frame = self.reduce_colors_kmeans(frame, max_colors=64)

        # 3. Pixelize with quantize
        frame = self.pixelize_with_quantize(frame, grid_size=2)

        # 4. Apply GAMEBOY palette
        frame = self.apply_gameboy_palette(frame)

        # 5. Cleanup pixels
        frame = self.cleanup_pixels(frame, threshold=0.02)

        return frame

    def process_schloss_mp4(self, input_path, output_path, max_frames=100):
        """Verarbeite Schlossgifdance.mp4 mit Direct Modern Pixel Art"""
        print(f"🏰 SCHLOSS DIRECT MODERN PIXEL ART PROCESSING")
        print("=" * 60)

        # Extract Frames from MP4
        frames, frame_duration = self.extract_frames_from_mp4(
            input_path, max_frames)
        if not frames:
            return False

        print(
            f"\n🎯 Processing {len(frames)} frames with Direct Modern Pixel Art...")
        print("🚀 Starting Schloss processing...\n")

        try:
            processed_frames = []
            total_frames = len(frames)
            start_time = time.time()

            for frame_num, frame in enumerate(frames):
                print(
                    f"🖼️ Processing Frame {frame_num + 1}/{total_frames}...", end=" ")

                # Apply Modern Pixel Art
                processed_frame = self.apply_modern_pixel_art(frame)
                processed_frames.append(processed_frame)

                elapsed = time.time() - start_time
                avg_time = elapsed / (frame_num + 1)
                eta = (total_frames - frame_num - 1) * avg_time
                progress = (frame_num + 1) / total_frames * 100

                print(f"✅ ({progress:.1f}% done, ETA: {eta:.1f}s)")

            # Save Modern Pixel Art GIF
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

            print(f"\n🎉 SCHLOSS DIRECT MODERN PIXEL ART COMPLETE!")
            print("=" * 60)
            print(f"✅ Output GIF: {output_path}")
            print(f"📊 Final Statistics:")
            print(f"   Total Frames: {len(processed_frames)}")
            print(f"   Processing Time: {total_time:.1f} seconds")
            print(
                f"   Average per Frame: {total_time/len(processed_frames):.2f} seconds")
            print(f"   Style: Modern Pixel Art (GAMEBOY Palette)")
            print(f"   Resolution: 512x512")
            print(f"   Frame Duration: {frame_duration}ms")
            print(f"   Matches Style: pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif")

            return True

        except Exception as e:
            print(f"❌ Schloss processing error: {e}")
            return False

        return False


def main():
    """Process Schlossgifdance.mp4 with Direct Modern Pixel Art"""
    print("🏰 SCHLOSS DIRECT MODERN PIXEL ART PROCESSOR")
    print("=" * 60)

    processor = SchlossDirectPixelProcessor()

    # Target Files
    input_file = "input/Schlossgifdance.mp4"
    output_file = "output/SCHLOSS_DIRECT_MODERN_PIXEL_ART.gif"

    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        return

    # Process mit max 100 Frames (für reasonable processing time)
    success = processor.process_schloss_mp4(
        input_file, output_file, max_frames=100)

    if success:
        print(
            f"\n🎉 SUCCESS! Direct Modern Pixel Art Schloss ready: {output_file}")
        print(f"📸 Style matches: pixel_modern_0ea53a3cdbfdcf14caf1c8cccdb60143.gif")
    else:
        print(f"\n❌ Processing failed.")


if __name__ == "__main__":
    main()
