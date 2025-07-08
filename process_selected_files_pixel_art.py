#!/usr/bin/env python3
"""
🎨 SELECTED FILES PIXEL ART PROCESSOR
===================================
Wendet den identifizierten Pixel Art Workflow auf die vom User
spezifizierten Dateien an.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time


class SelectedFilesPixelArtProcessor:
    def __init__(self):
        # GAMEBOY Palette (aus dem identifizierten Workflow)
        self.gameboy_palette = [
            (15, 56, 15),      # Dunkelgrün
            (48, 98, 48),      # Mittelgrün
            (139, 172, 15),    # Hellgrün
            (155, 188, 15)     # Sehr hellgrün
        ]

        # Target files liste (vom User spezifiziert)
        self.target_files = [
            "input/1938caaca4055d456a9c12ef8648a057_fast_transparent_converted.gif",
            "input/2609a4ee571128f2079373b8d7b0a1a0_fast_transparent_converted.gif",
            "input/antrieb_fast_transparent_converted.gif",
            "input/ebb946e99e5ff654fdaf45112ddac4c7_fast_transparent_converted.gif",
            "input/R (3)_fast_transparent_converted.gif",
            "input/spiral_fast_transparent_converted.gif",
            "input/P9vTI0_fast_transparent_converted.gif",
            "input/tableware-milk-fill-only-34_fast_transparent_converted.gif",
            "input/tumblr_3c7ddc41f9d033983af0359360d773cf_38fa1d69_540_fast_transparent_converted.gif",
            "input/0af40433ddb755bfee5a1738717c7028_fast_transparent_converted.gif",
            "input/uJ1Dg2_fast_transparent_converted.gif",
            "input/R (2)_fast_transparent_converted.gif",
            "input/yv80rldigcu61_fast_transparent_converted.gif",
            "input/giphy_fast_transparent_converted.gif",
            "input/ffcb41ab727135955c859e88bc286c54_fast_transparent_converted.gif",
            "input/tenor_fast_transparent_converted.gif",
            "input/tumblr_inline_nfpj8uucP11s6lw3t540_fast_transparent_converted.gif",
            "input/gym-roshi_2_fast_transparent_converted.gif",
            "input/0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif",
            "input/Schlossgifdance.mp4",
            "input/milchzoom.mp4",
            "input/837b898b4d1eb49036dfce89c30cba59_fast_transparent_converted.gif"
        ]

        print("🎨 Selected Files Pixel Art Processor initialized")
        print(f"📁 {len(self.target_files)} Dateien zu verarbeiten")

    def extract_frames_from_gif(self, gif_path):
        """Extrahiere Frames aus GIF"""
        try:
            gif = Image.open(gif_path)
            frames = []
            frame_duration = gif.info.get('duration', 100)

            frame_count = 0
            while True:
                try:
                    frame = gif.copy().convert('RGBA')
                    frames.append(frame)
                    frame_count += 1
                    gif.seek(frame_count)
                except EOFError:
                    break

            print(
                f"   📹 Extracted {len(frames)} frames, duration: {frame_duration}ms")
            return frames, frame_duration

        except Exception as e:
            print(f"   ❌ GIF extraction error: {e}")
            return None, None

    def extract_frames_from_mp4(self, mp4_path, max_frames=100):
        """Extrahiere Frames aus MP4"""
        try:
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_count += 1

            cap.release()
            frame_duration = int(1000 / fps) if fps > 0 else 100

            print(f"   📹 Extracted {len(frames)} frames from MP4, FPS: {fps}")
            return frames, frame_duration

        except Exception as e:
            print(f"   ❌ MP4 extraction error: {e}")
            return None, None

    def apply_pixel_art_workflow(self, frame):
        """Kompletter Pixel Art Workflow"""
        # 1. Resize zu 512x512
        frame = frame.resize((512, 512), Image.LANCZOS)

        # 2. Color reduction mit K-Means
        img_array = np.array(frame.convert('RGB'))
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 64, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        reduced_pixels = centers[labels.flatten()]
        reduced_img = reduced_pixels.reshape(original_shape)
        frame = Image.fromarray(reduced_img)

        # 3. Pixelize mit quantize
        small_size = (frame.size[0] // 8, frame.size[1] // 8)
        small_img = frame.resize(small_size, Image.NEAREST)
        quantized = small_img.quantize(colors=128, method=Image.MAXCOVERAGE)
        frame = quantized.resize((512, 512), Image.NEAREST).convert('RGB')

        # 4. Apply gameboy palette
        img_array = np.array(frame)
        pixels = img_array.reshape(-1, 3)
        palette_array = np.array(self.gameboy_palette)

        distances = np.sqrt(
            np.sum((pixels[:, None, :] - palette_array[None, :, :]) ** 2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        new_pixels = palette_array[closest_indices]
        new_img_array = new_pixels.reshape(img_array.shape).astype(np.uint8)
        frame = Image.fromarray(new_img_array)

        # 5. Upscale 4x
        new_size = (frame.size[0] * 4, frame.size[1] * 4)
        frame = frame.resize(new_size, Image.NEAREST)

        return frame

    def create_video(self, frames, output_path, fps=24):
        """Video erstellen mit H.264"""
        if not frames:
            return False

        try:
            height, width = np.array(frames[0]).shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                frame_array = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            return True
        except Exception as e:
            print(f"   ❌ Video creation error: {e}")
            return False

    def process_file(self, input_path, output_dir):
        """Verarbeite einzelne Datei"""
        file_path = Path(input_path)

        if not file_path.exists():
            print(f"❌ Datei nicht gefunden: {input_path}")
            return False

        filename = file_path.stem
        if filename.endswith("_fast_transparent_converted"):
            filename = filename.replace("_fast_transparent_converted", "")

        output_path = output_dir / f"{filename}_pixelart.mp4"

        print(f"🔄 Verarbeite: {file_path.name}")

        # Extrahiere Frames
        if file_path.suffix.lower() == '.gif':
            frames, frame_duration = self.extract_frames_from_gif(input_path)
            fps = 1000 / frame_duration if frame_duration > 0 else 24
        elif file_path.suffix.lower() == '.mp4':
            frames, frame_duration = self.extract_frames_from_mp4(input_path)
            fps = 1000 / frame_duration if frame_duration > 0 else 24
        else:
            print(f"   ❌ Unsupported file type: {file_path.suffix}")
            return False

        if not frames:
            return False

        # Pixel Art Processing
        print(f"   🎨 Applying Pixel Art Workflow to {len(frames)} frames...")
        processed_frames = []

        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"      Frame {i+1}/{len(frames)}")

            processed_frame = self.apply_pixel_art_workflow(frame)
            processed_frames.append(processed_frame)

        # Video erstellen
        print(f"   💾 Creating video: {output_path.name}")
        success = self.create_video(processed_frames, str(output_path), fps)

        if success:
            print(f"   ✅ Success: {output_path}")
            return True
        else:
            print(f"   ❌ Failed: {output_path}")
            return False

    def process_all_files(self):
        """Verarbeite alle spezifizierten Dateien"""
        print("🎬 SELECTED FILES PIXEL ART PROCESSING")
        print("=" * 60)

        # Output-Verzeichnis erstellen
        output_dir = Path("output/selected_files_pixel_art")
        output_dir.mkdir(parents=True, exist_ok=True)

        successful = 0
        failed = 0
        start_time = time.time()

        for i, file_path in enumerate(self.target_files):
            print(f"\n📁 [{i+1}/{len(self.target_files)}] Processing...")

            success = self.process_file(file_path, output_dir)

            if success:
                successful += 1
            else:
                failed += 1

            # Progress
            progress = (i + 1) / len(self.target_files) * 100
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * \
                (len(self.target_files) - i - 1) if i > 0 else 0

            print(f"📊 Progress: {progress:.1f}% | ETA: {eta:.1f}s")

        total_time = time.time() - start_time

        print(f"\n🎉 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successful: {successful} files")
        print(f"❌ Failed: {failed} files")
        print(f"⏱️ Total Time: {total_time:.1f} seconds")
        print(f"📂 Output Directory: {output_dir}")
        print("=" * 60)

        return successful, failed


def main():
    """Process selected files with Pixel Art Workflow"""
    print("🎨 SELECTED FILES PIXEL ART PROCESSOR")
    print("=" * 60)

    processor = SelectedFilesPixelArtProcessor()

    # Zeige Dateiliste
    print(f"\n📋 Files to process:")
    for i, file_path in enumerate(processor.target_files[:5]):
        print(f"   {i+1}. {Path(file_path).name}")
    if len(processor.target_files) > 5:
        print(f"   ... und {len(processor.target_files) - 5} weitere")

    # Starte Verarbeitung
    successful, failed = processor.process_all_files()

    if successful > 0:
        print(f"\n🎉 SUCCESS! {successful} Pixel Art Videos erstellt!")
        print("📁 Output: output/selected_files_pixel_art/")
    if failed > 0:
        print(f"⚠️ {failed} Dateien fehlgeschlagen")


if __name__ == "__main__":
    main()
