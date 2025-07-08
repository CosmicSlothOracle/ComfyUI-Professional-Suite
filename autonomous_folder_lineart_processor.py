#!/usr/bin/env python3
"""
🎬 AUTONOMOUS FOLDER LINE ART PROCESSOR
======================================
Verarbeitet ALLE Videos/GIFs im Input-Ordner autonom zu Line Art Animationen
Intelligente Erkennung und Batch-Processing für große Mengen
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import time
import os
import glob


class AutonomousFolderLineArtProcessor:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output/COMPLETE_FOLDER_LINE_ART")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unterstützte Formate
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        self.gif_extensions = ['.gif']
        self.supported_extensions = self.video_extensions + self.gif_extensions

        print("🎬 AUTONOMOUS FOLDER LINE ART PROCESSOR")
        print("=" * 60)
        print("🎯 Target: COMPLETE INPUT FOLDER → Professional Line Art")
        print("⚙️  Method: Direct OpenCV + PIL Implementation")
        print("🚀 Mode: Vollständig autonom - Complete Folder Processing")
        print(f"📁 Source: {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        print()

    def scan_input_folder(self):
        """Scanne Input-Ordner nach unterstützten Dateien"""
        print("🔍 Scanning input folder for media files...")

        all_files = []

        # Finde alle unterstützten Dateien
        for extension in self.supported_extensions:
            pattern = str(self.input_dir / f"*{extension}")
            files = glob.glob(pattern, recursive=False)
            all_files.extend(files)

        # Sortiere und bereinige
        media_files = []
        for file_path in all_files:
            path = Path(file_path)
            if path.exists() and path.is_file():
                media_files.append(str(path))

        # Nach Dateigröße sortieren (kleinere zuerst für bessere Performance)
        media_files.sort(key=lambda x: Path(x).stat().st_size)

        print(f"📊 Found {len(media_files)} media files:")

        # Gruppiere nach Typ
        videos = [f for f in media_files if Path(
            f).suffix.lower() in self.video_extensions]
        gifs = [f for f in media_files if Path(
            f).suffix.lower() in self.gif_extensions]

        print(f"   🎬 Videos: {len(videos)}")
        print(f"   🎨 GIFs: {len(gifs)}")

        # Zeige erste 10 und letzte 5 Dateien
        if len(media_files) > 15:
            print(f"\n📋 Sample files (showing 10 of {len(media_files)}):")
            for i, filepath in enumerate(media_files[:10]):
                size_mb = Path(filepath).stat().st_size / (1024*1024)
                print(f"   {i+1:2d}. {Path(filepath).name} ({size_mb:.1f}MB)")
            print(f"   ... and {len(media_files)-10} more files")
        else:
            print(f"\n📋 All files:")
            for i, filepath in enumerate(media_files):
                size_mb = Path(filepath).stat().st_size / (1024*1024)
                print(f"   {i+1:2d}. {Path(filepath).name} ({size_mb:.1f}MB)")

        return media_files

    def get_safe_filename(self, filepath):
        """Erstelle sicheren Dateinamen für Output"""
        filename = Path(filepath).stem
        # Entferne problematische Zeichen
        safe_name = "".join(c for c in filename if c.isalnum()
                            or c in (' ', '-', '_', '.')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        # Kürze sehr lange Namen
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        return safe_name

    def estimate_processing_time(self, media_files):
        """Schätze Processing-Zeit"""
        total_size = sum(Path(f).stat().st_size for f in media_files)
        total_mb = total_size / (1024*1024)

        # Schätzung basierend auf Erfahrung: ~50MB pro Minute
        estimated_minutes = total_mb / 50

        print(f"📊 Processing estimation:")
        print(f"   Total size: {total_mb:.1f} MB")
        print(f"   Estimated time: {estimated_minutes:.1f} minutes")
        print(
            f"   Average per file: {estimated_minutes*60/len(media_files):.1f} seconds")

        return estimated_minutes

    def extract_frames_from_media(self, filepath, max_frames=None):
        """Extrahiere Frames aus Video oder GIF"""
        file_size = Path(filepath).stat().st_size / (1024*1024)

        # Adaptive max_frames basierend auf Dateigröße
        if max_frames is None:
            if file_size < 1:  # < 1MB
                max_frames = 30
            elif file_size < 5:  # < 5MB
                max_frames = 50
            elif file_size < 10:  # < 10MB
                max_frames = 75
            else:  # >= 10MB
                max_frames = 100

        try:
            cap = cv2.VideoCapture(str(filepath))

            # Media Info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Fallback für GIFs mit fps=0
            if fps == 0:
                fps = 10  # Standard GIF framerate

            process_frames = min(
                total_frames, max_frames) if total_frames > 0 else max_frames

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

            cap.release()
            return frames, fps

        except Exception as e:
            print(f"❌ Frame extraction error for {Path(filepath).name}: {e}")
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

    def create_output_from_frames(self, frames, fps, output_path, is_gif=False):
        """Erstelle Output (Video oder GIF) aus Frames"""
        if not frames:
            return False

        if is_gif:
            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(50, int(1000/fps)),  # Min 50ms per frame
                loop=0,
                optimize=True
            )
        else:
            # Save as MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames[0].height, frames[0].width

            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            for frame in frames:
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(cv_frame)

            out.release()

        return True

    def process_single_file(self, filepath, file_index, total_files):
        """Verarbeite eine einzelne Datei mit Progress"""
        file_path = Path(filepath)
        safe_name = self.get_safe_filename(filepath)

        print(f"\n📁 [{file_index}/{total_files}] Processing: {file_path.name}")
        print("-" * 50)

        start_time = time.time()

        # 1. Extract Frames
        frames, fps = self.extract_frames_from_media(filepath)
        if not frames:
            print(f"❌ Failed to extract frames")
            return False

        print(f"📹 Extracted {len(frames)} frames")
        print(f"🎨 Converting to Line Art...")

        # 2. Process Frames zu Line Art (mit reduziertem Logging für Performance)
        processed_frames = []
        total_frames = len(frames)

        for frame_num, frame in enumerate(frames):
            # Nur jede 10. Frame loggen für Performance
            if frame_num % 10 == 0:
                progress = (frame_num + 1) / total_frames * 100
                print(
                    f"   🖼️  Frame {frame_num + 1}/{total_frames} ({progress:.0f}%)")

            line_art_frame = self.process_frame_to_line_art(frame)
            processed_frames.append(line_art_frame)

        # 3. Create Output
        file_extension = file_path.suffix.lower()
        is_gif = file_extension in self.gif_extensions

        if is_gif:
            output_path = self.output_dir / f"{safe_name}_LINE_ART.gif"
        else:
            output_path = self.output_dir / f"{safe_name}_LINE_ART.mp4"

        success = self.create_output_from_frames(
            processed_frames, fps, output_path, is_gif)

        processing_time = time.time() - start_time

        if success:
            file_size = output_path.stat().st_size / (1024*1024)
            print(f"✅ SUCCESS: {output_path.name} ({file_size:.1f}MB)")
            print(f"   ⏱️  Time: {processing_time:.1f}s")
            print(f"   📊 Frames: {len(processed_frames)}")
            return True
        else:
            print(f"❌ FAILED: {file_path.name}")
            return False

    def process_complete_folder(self):
        """Verarbeite kompletten Input-Ordner"""
        print("🚀 STARTING COMPLETE FOLDER LINE ART CONVERSION")
        print("=" * 60)

        # 1. Scan Folder
        media_files = self.scan_input_folder()

        if not media_files:
            print("❌ No media files found in input folder!")
            return 0, 0

        # 2. Estimate Processing Time
        estimated_time = self.estimate_processing_time(media_files)

        input(
            f"\n⏳ Processing {len(media_files)} files (est. {estimated_time:.1f} min). Press ENTER to continue...")

        total_start_time = time.time()
        successful_files = []
        failed_files = []

        # 3. Process each file
        for i, filepath in enumerate(media_files, 1):
            try:
                success = self.process_single_file(
                    filepath, i, len(media_files))

                if success:
                    successful_files.append(filepath)
                else:
                    failed_files.append(filepath)

            except Exception as e:
                print(f"❌ Error processing {Path(filepath).name}: {e}")
                failed_files.append(filepath)

            # Progress Update
            elapsed = time.time() - total_start_time
            remaining_files = len(media_files) - i
            avg_time = elapsed / i
            eta = remaining_files * avg_time

            print(f"\n📊 Progress: {i}/{len(media_files)} files completed")
            print(f"⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")

        # 4. Final Report
        total_time = time.time() - total_start_time

        print(f"\n🎉 COMPLETE FOLDER PROCESSING FINISHED!")
        print("=" * 60)
        print(f"✅ Successful: {len(successful_files)} files")
        print(f"❌ Failed: {len(failed_files)} files")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"📁 Output Directory: {self.output_dir}")

        if successful_files:
            print(
                f"\n🎨 Successfully created {len(successful_files)} Line Art animations!")

        if failed_files:
            print(f"\n⚠️  Failed files:")
            for filepath in failed_files[:10]:  # Zeige nur erste 10
                print(f"   ❌ {Path(filepath).name}")
            if len(failed_files) > 10:
                print(f"   ... and {len(failed_files)-10} more")

        return len(successful_files), len(failed_files)


def main():
    """Hauptfunktion für Complete Folder Processing"""
    print("🎬 AUTONOMOUS FOLDER LINE ART PROCESSOR")
    print("=" * 60)
    print("🎯 Target: COMPLETE INPUT FOLDER → Professional Line Art")
    print("⚙️  Method: Direct OpenCV + PIL (No ComfyUI needed)")
    print("🚀 Mode: Completely Autonomous Folder Processing")
    print()

    processor = AutonomousFolderLineArtProcessor()

    # Process complete folder
    successful, failed = processor.process_complete_folder()

    print(f"\n🎉 FOLDER PROCESSING SUMMARY:")
    print(f"✅ {successful} files successfully converted to Line Art")
    print(f"❌ {failed} files failed")
    print(f"📁 All outputs in: {processor.output_dir}")
    print(f"\n🎨 Complete Input Folder processed to Professional Line Art! ✨")


if __name__ == "__main__":
    main()
