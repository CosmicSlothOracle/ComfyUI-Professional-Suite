#!/usr/bin/env python3
"""
🎬 AUTONOMOUS COMPLETE FOLDER PROCESSOR
=====================================
Verarbeitet ALLE Videos/GIFs im Input-Ordner zu Line Art Animationen
Vollständig autonom - Complete Folder Processing für große Mengen
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import time
import os
import glob


class CompleteInputFolderProcessor:
    def __init__(self):
        self.input_dir = Path("input")
        self.output_dir = Path("output/COMPLETE_INPUT_FOLDER_LINE_ART")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unterstützte Formate
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        self.gif_extensions = ['.gif']
        self.supported_extensions = self.video_extensions + self.gif_extensions

        print("🎬 COMPLETE INPUT FOLDER LINE ART PROCESSOR")
        print("=" * 60)
        print("🎯 Target: ENTIRE INPUT FOLDER → Professional Line Art")
        print("⚙️  Method: Autonomous OpenCV + PIL Processing")
        print("📁 Processing ALL video/GIF files in input folder")
        print()

    def scan_all_media_files(self):
        """Scanne Input-Ordner nach ALLEN unterstützten Dateien"""
        print("🔍 Scanning complete input folder...")

        all_files = []

        # Finde alle unterstützten Dateien
        for extension in self.supported_extensions:
            pattern = str(self.input_dir / f"*{extension}")
            files = glob.glob(pattern, recursive=False)
            all_files.extend(files)

        # Zusätzlich: Case-insensitive search
        for extension in self.supported_extensions:
            pattern = str(self.input_dir / f"*{extension.upper()}")
            files = glob.glob(pattern, recursive=False)
            all_files.extend(files)

        # Entferne Duplikate und validiere
        unique_files = []
        for file_path in set(all_files):  # set() entfernt Duplikate
            path = Path(file_path)
            if path.exists() and path.is_file():
                unique_files.append(str(path))

        # Nach Name sortieren für bessere Übersicht
        unique_files.sort()

        print(f"📊 Found {len(unique_files)} media files in input folder")

        # Gruppiere nach Typ
        videos = [f for f in unique_files if Path(
            f).suffix.lower() in self.video_extensions]
        gifs = [f for f in unique_files if Path(
            f).suffix.lower() in self.gif_extensions]

        print(f"   🎬 Videos: {len(videos)}")
        print(f"   🎨 GIFs: {len(gifs)}")

        # Zeige Dateigrößen-Statistik
        total_size = sum(Path(f).stat().st_size for f in unique_files)
        print(f"   📦 Total size: {total_size/(1024*1024*1024):.2f} GB")

        return unique_files

    def get_adaptive_frame_limit(self, filepath):
        """Bestimme intelligente Frame-Limits basierend auf Dateigröße"""
        file_size_mb = Path(filepath).stat().st_size / (1024*1024)

        if file_size_mb < 1:      # < 1MB
            return 25
        elif file_size_mb < 3:    # 1-3MB
            return 40
        elif file_size_mb < 7:    # 3-7MB
            return 60
        elif file_size_mb < 15:   # 7-15MB
            return 80
        else:                     # > 15MB
            return 100

    def extract_frames_smart(self, filepath):
        """Smart Frame Extraction mit adaptiven Limits"""
        max_frames = self.get_adaptive_frame_limit(filepath)

        try:
            cap = cv2.VideoCapture(str(filepath))

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # GIF Fallback
            if fps == 0 or fps is None:
                fps = 10

            frames = []
            frame_count = 0

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
                frame_count += 1

            cap.release()
            return frames, fps

        except Exception as e:
            print(f"   ❌ Frame extraction failed: {e}")
            return None, None

    def convert_to_line_art(self, image):
        """Professionelle Line Art Konversion"""
        # Standard-Größe für konsistente Verarbeitung
        target_size = (512, 512)
        image = image.resize(target_size, Image.LANCZOS)

        # OpenCV Konversion
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Multi-step Line Art Process
        # 1. Noise Reduction
        denoised = cv2.medianBlur(cv_image, 5)

        # 2. Grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        # 3. Edge Detection kombiniert
        edges = cv2.Canny(gray, 50, 150)

        # 4. Adaptive Threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )

        # 5. Kombiniere beide Methoden
        combined = cv2.bitwise_or(adaptive, edges)

        # 6. Morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # 7. Invert für schwarze Linien
        final = cv2.bitwise_not(cleaned)

        # Zurück zu PIL
        line_art_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(line_art_rgb)

    def process_single_media_file(self, filepath, index, total):
        """Verarbeite eine einzelne Media-Datei"""
        filename = Path(filepath).name
        safe_name = Path(filepath).stem.replace(' ', '_')[:50]  # Sichere Namen

        print(f"\n🎬 [{index}/{total}] {filename}")
        print(f"   📏 {Path(filepath).stat().st_size/(1024*1024):.1f}MB")

        start_time = time.time()

        # 1. Frame Extraction
        frames, fps = self.extract_frames_smart(filepath)
        if not frames:
            print(f"   ❌ Failed to extract frames")
            return False

        print(f"   📹 Extracted {len(frames)} frames")

        # 2. Line Art Conversion
        processed_frames = []
        for i, frame in enumerate(frames):
            if i % 20 == 0:  # Nur jede 20. Frame loggen
                print(f"   🎨 Processing frame {i+1}/{len(frames)}")

            line_art = self.convert_to_line_art(frame)
            processed_frames.append(line_art)

        # 3. Output Creation
        extension = Path(filepath).suffix.lower()
        if extension in self.gif_extensions:
            output_path = self.output_dir / f"{safe_name}_line_art.gif"
            # Save GIF
            processed_frames[0].save(
                output_path,
                save_all=True,
                append_images=processed_frames[1:],
                duration=max(50, int(1000/fps)),
                loop=0,
                optimize=True
            )
        else:
            output_path = self.output_dir / f"{safe_name}_line_art.mp4"
            # Save MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = 512, 512
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            for frame in processed_frames:
                cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                out.write(cv_frame)
            out.release()

        processing_time = time.time() - start_time
        output_size = output_path.stat().st_size / (1024*1024)

        print(
            f"   ✅ Complete: {output_path.name} ({output_size:.1f}MB, {processing_time:.1f}s)")
        return True

    def process_complete_input_folder(self):
        """Verarbeite den kompletten Input-Ordner"""
        print("🚀 STARTING COMPLETE INPUT FOLDER PROCESSING")
        print("=" * 60)

        # 1. Finde alle Media-Dateien
        media_files = self.scan_all_media_files()

        if not media_files:
            print("❌ No media files found!")
            return

        # 2. Schätze Verarbeitungszeit
        total_size_gb = sum(
            Path(f).stat().st_size for f in media_files) / (1024*1024*1024)
        estimated_hours = total_size_gb * 0.5  # Rough estimate

        print(f"\n📊 Processing Overview:")
        print(f"   Files to process: {len(media_files)}")
        print(f"   Total data: {total_size_gb:.2f} GB")
        print(f"   Estimated time: {estimated_hours:.1f} hours")

        # Bestätigung
        proceed = input(
            f"\n⚠️  This will process {len(media_files)} files. Continue? (y/N): ").lower()
        if proceed != 'y':
            print("❌ Processing cancelled.")
            return

        # 3. Verarbeitung starten
        successful = 0
        failed = 0
        start_time = time.time()

        for i, filepath in enumerate(media_files, 1):
            try:
                success = self.process_single_media_file(
                    filepath, i, len(media_files))
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed += 1

            # Progress Update
            elapsed = time.time() - start_time
            if i % 10 == 0:  # Jede 10. Datei
                remaining = len(media_files) - i
                avg_time = elapsed / i
                eta_hours = (remaining * avg_time) / 3600
                print(f"\n📊 Progress Update: {i}/{len(media_files)} files")
                print(
                    f"   ⏱️  Elapsed: {elapsed/3600:.1f}h, ETA: {eta_hours:.1f}h")

        # 4. Final Report
        total_time = time.time() - start_time

        print(f"\n🎉 COMPLETE INPUT FOLDER PROCESSING FINISHED!")
        print("=" * 60)
        print(f"✅ Successfully processed: {successful} files")
        print(f"❌ Failed: {failed} files")
        print(f"⏱️  Total processing time: {total_time/3600:.1f} hours")
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"\n🎨 Complete input folder converted to Line Art! ✨")

        return successful, failed


def main():
    """Hauptfunktion für Complete Input Folder Processing"""
    print("🎬 AUTONOMOUS COMPLETE INPUT FOLDER PROCESSOR")
    print("=" * 60)

    processor = CompleteInputFolderProcessor()

    # Verarbeite kompletten Input-Ordner
    processor.process_complete_input_folder()


if __name__ == "__main__":
    main()
