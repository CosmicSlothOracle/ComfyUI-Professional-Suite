#!/usr/bin/env python3
"""
🎬 AUTONOMOUS BATCH LINE ART PROCESSOR
=====================================
Verarbeitet alle angegebenen Videos/GIFs autonom zu Line Art Animationen
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import time
import os


class AutonomousBatchLineArtProcessor:
    def __init__(self):
        # Liste der zu verarbeitenden Dateien
        self.input_files = [
            "input/12 - Kopie.gif",
            "input/0ea53a3cdbfdcf14caf1c8cccdb60143_fast_transparent_converted.gif",
            "input/Schlossgifdance.mp4",
            "input/milchzoom.mp4",
            "input/giphy - Kopie.gif",
            "input/333 - Kopie.gif"
        ]

        self.output_dir = Path("output/BATCH_LINE_ART_ANIMATIONS")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("🎬 AUTONOMOUS BATCH LINE ART PROCESSOR")
        print("=" * 60)
        print("🎯 Target: 6 Videos/GIFs → Professional Line Art")
        print("⚙️  Method: Direct OpenCV + PIL Implementation")
        print("🚀 Mode: Vollständig autonom - Batch Processing")
        print()

    def get_safe_filename(self, filepath):
        """Erstelle sicheren Dateinamen für Output"""
        filename = Path(filepath).stem
        # Entferne problematische Zeichen
        safe_name = "".join(c for c in filename if c.isalnum()
                            or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        return safe_name

    def extract_frames_from_media(self, filepath, max_frames=100):
        """Extrahiere Frames aus Video oder GIF"""
        print(f"📹 Extracting frames from: {Path(filepath).name}")

        if not Path(filepath).exists():
            print(f"❌ File not found: {filepath}")
            return None, None

        try:
            cap = cv2.VideoCapture(str(filepath))

            # Media Info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Fallback für GIFs mit fps=0
            if fps == 0:
                fps = 10  # Standard GIF framerate

            duration = total_frames / fps if fps > 0 else total_frames * 0.1

            print(f"   📊 FPS: {fps}")
            print(f"   📊 Total Frames: {total_frames}")
            print(f"   📊 Duration: {duration:.1f}s")

            if max_frames:
                process_frames = min(total_frames, max_frames)
                print(f"   📊 Processing: {process_frames} frames (limited)")
            else:
                process_frames = total_frames

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= process_frames:
                    break

                # OpenCV BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_count += 1

                if frame_count % 20 == 0:
                    print(
                        f"   📹 Extracted: {frame_count}/{process_frames} frames")

            cap.release()

            print(f"✅ Extracted {len(frames)} frames")

            return frames, fps

        except Exception as e:
            print(f"❌ Frame extraction error: {e}")
            return None, None

    def advanced_line_art_extraction(self, image):
        """Erweiterte Line Art Extraktion mit mehreren Methoden"""
        # Konvertiere zu OpenCV Format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 1. Gaussian Blur für Noise Reduction
        blurred = cv2.GaussianBlur(cv_image, (3, 3), 0)

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 3. Adaptive Threshold für saubere Linien
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )

        # 4. Canny Edge Detection
        edges = cv2.Canny(gray, 50, 150)

        # 5. Kombiniere Adaptive Threshold und Canny
        combined = cv2.bitwise_or(adaptive_thresh, edges)

        # 6. Morphological Operations für saubere Linien
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # 7. Invert für schwarze Linien auf weißem Hintergrund
        inverted = cv2.bitwise_not(cleaned)

        # Convert zurück zu PIL RGB
        line_art = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(line_art)

    def enhance_line_art(self, image):
        """Verbessere Line Art Quality"""
        # 1. Contrast Enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.3)

        # 2. Sharpness Enhancement
        enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = enhancer.enhance(1.2)

        # 3. Slight Smoothing
        smoothed = sharpened.filter(ImageFilter.SMOOTH_MORE)

        return smoothed

    def process_frame_to_line_art(self, frame):
        """Verarbeite einzelnen Frame zu Line Art"""
        # 1. Resize für konsistente Verarbeitung
        target_size = (512, 512)
        frame = frame.resize(target_size, Image.LANCZOS)

        # 2. Line Art Extraktion
        line_art = self.advanced_line_art_extraction(frame)

        # 3. Enhancement
        enhanced = self.enhance_line_art(line_art)

        return enhanced

    def create_video_from_frames(self, frames, fps, output_path):
        """Erstelle Video aus Frames"""
        if not frames:
            print("❌ No frames to process")
            return False

        # Video Writer Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].height, frames[0].width

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for i, frame in enumerate(frames):
            # PIL → OpenCV (RGB → BGR)
            cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

        out.release()
        return True

    def create_gif_from_frames(self, frames, output_path, duration=100):
        """Erstelle GIF aus Frames"""
        if not frames:
            return False

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        return True

    def process_single_file(self, filepath, max_frames=100):
        """Verarbeite eine einzelne Datei"""
        file_path = Path(filepath)
        safe_name = self.get_safe_filename(filepath)

        print(f"\n🎬 PROCESSING: {file_path.name}")
        print("-" * 50)

        start_time = time.time()

        # 1. Extract Frames
        frames, fps = self.extract_frames_from_media(filepath, max_frames)
        if not frames:
            print(f"❌ Failed to extract frames from {file_path.name}")
            return False

        print(f"🎨 Processing {len(frames)} frames to Line Art...")

        # 2. Process Frames zu Line Art
        processed_frames = []
        total_frames = len(frames)

        for frame_num, frame in enumerate(frames):
            if frame_num % 20 == 0:
                progress = (frame_num + 1) / total_frames * 100
                print(
                    f"   🖼️  Processing: {frame_num + 1}/{total_frames} ({progress:.1f}%)")

            line_art_frame = self.process_frame_to_line_art(frame)
            processed_frames.append(line_art_frame)

        # 3. Create Output
        file_extension = file_path.suffix.lower()

        if file_extension in ['.mp4', '.avi', '.mov']:
            # Save as MP4
            output_path = self.output_dir / f"{safe_name}_LINE_ART.mp4"
            success = self.create_video_from_frames(
                processed_frames, fps, output_path)
        else:
            # Save as GIF
            output_path = self.output_dir / f"{safe_name}_LINE_ART.gif"
            success = self.create_gif_from_frames(
                processed_frames, output_path, duration=100)

        processing_time = time.time() - start_time

        if success:
            print(f"✅ SUCCESS: {output_path.name}")
            print(f"   ⏱️  Time: {processing_time:.1f}s")
            print(f"   📊 Frames: {len(processed_frames)}")
            print(f"   🎨 Style: Professional Line Art")
            return True
        else:
            print(f"❌ FAILED: {file_path.name}")
            return False

    def process_all_files(self):
        """Verarbeite alle Dateien im Batch"""
        print("🚀 STARTING BATCH LINE ART CONVERSION")
        print("=" * 60)

        total_start_time = time.time()
        successful_files = []
        failed_files = []

        # Check welche Dateien existieren
        existing_files = []
        for filepath in self.input_files:
            if Path(filepath).exists():
                existing_files.append(filepath)
                print(f"✅ Found: {Path(filepath).name}")
            else:
                print(f"⚠️  Not found: {Path(filepath).name}")
                failed_files.append(filepath)

        print(f"\n📊 Processing {len(existing_files)} files...")

        # Verarbeite jede Datei
        for i, filepath in enumerate(existing_files, 1):
            print(
                f"\n📁 [{i}/{len(existing_files)}] Processing: {Path(filepath).name}")

            # Angepasste max_frames je nach Dateityp
            if Path(filepath).suffix.lower() in ['.gif']:
                max_frames = 50  # GIFs können viele Frames haben
            else:
                max_frames = 100  # Videos

            success = self.process_single_file(filepath, max_frames)

            if success:
                successful_files.append(filepath)
            else:
                failed_files.append(filepath)

        # Final Report
        total_time = time.time() - total_start_time

        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"✅ Successful: {len(successful_files)} files")
        print(f"❌ Failed: {len(failed_files)} files")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print(f"📁 Output Directory: {self.output_dir}")

        if successful_files:
            print(f"\n📋 Successfully processed:")
            for filepath in successful_files:
                safe_name = self.get_safe_filename(filepath)
                extension = ".mp4" if Path(filepath).suffix.lower() in [
                    '.mp4', '.avi', '.mov'] else ".gif"
                output_name = f"{safe_name}_LINE_ART{extension}"
                print(f"   ✅ {Path(filepath).name} → {output_name}")

        if failed_files:
            print(f"\n⚠️  Failed files:")
            for filepath in failed_files:
                print(f"   ❌ {Path(filepath).name}")

        print(f"\n🎬 All Line Art animations ready in: {self.output_dir}")

        return len(successful_files), len(failed_files)


def main():
    """Hauptfunktion für Batch Processing"""
    print("🎬 AUTONOMOUS BATCH LINE ART PROCESSOR")
    print("=" * 60)
    print("🎯 Target: 6 Videos/GIFs → Professional Line Art")
    print("⚙️  Method: Direct OpenCV + PIL (No ComfyUI needed)")
    print("🚀 Mode: Completely Autonomous Batch Processing")
    print()

    processor = AutonomousBatchLineArtProcessor()

    # Process all files
    successful, failed = processor.process_all_files()

    print(f"\n🎉 BATCH PROCESSING SUMMARY:")
    print(f"✅ {successful} files successfully converted to Line Art")
    print(f"❌ {failed} files failed")
    print(f"📁 All outputs in: {processor.output_dir}")
    print(f"\n🎨 Professional Line Art animations ready! ✨")


if __name__ == "__main__":
    main()
