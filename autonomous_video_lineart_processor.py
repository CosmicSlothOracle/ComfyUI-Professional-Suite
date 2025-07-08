#!/usr/bin/env python3
"""
🎬 AUTONOMOUS VIDEO-TO-LINE ART PROCESSOR
========================================
Kompletter autonomer Video-zu-Line Art Konverter
OHNE ComfyUI Interface - direkte Python Implementation
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import time
import subprocess
import os


class AutonomousVideoLineArtProcessor:
    def __init__(self):
        self.input_video = "input/Schlossgifdance.mp4"
        self.output_path = "output/AUTONOMOUS_LINE_ART_ANIMATION.mp4"

        print("🎬 AUTONOMOUS VIDEO-TO-LINE ART PROCESSOR")
        print("=" * 60)
        print("🎯 Target: Schlossgifdance.mp4")
        print("🎨 Style: Professional Line Art")
        print("⚙️  Method: Direct OpenCV + PIL Implementation")
        print("🚀 Mode: Vollständig autonom")
        print()

    def extract_frames_from_video(self, max_frames=None):
        """Extrahiere Frames aus Video"""
        print("📹 Extracting frames from video...")

        if not Path(self.input_video).exists():
            print(f"❌ Video not found: {self.input_video}")
            return None, None

        try:
            cap = cv2.VideoCapture(self.input_video)

            # Video Info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            print(f"   📊 FPS: {fps}")
            print(f"   📊 Total Frames: {total_frames}")
            print(f"   📊 Duration: {duration:.1f}s")

            if max_frames:
                total_frames = min(total_frames, max_frames)
                print(f"   📊 Processing: {total_frames} frames (limited)")

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
                    print(
                        f"   📹 Extracted: {frame_count}/{total_frames} frames")

            cap.release()

            # Frame Duration für Output Video
            frame_duration = 1.0 / fps

            print(f"✅ Extracted {len(frames)} frames")
            print(f"📊 Frame Duration: {frame_duration:.3f}s")

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
        print(f"🎬 Creating video: {output_path}")

        if not frames:
            print("❌ No frames to process")
            return False

        # Video Writer Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].height, frames[0].width

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, frame in enumerate(frames):
            # PIL → OpenCV (RGB → BGR)
            cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(cv_frame)

            if (i + 1) % 30 == 0:
                print(f"   📹 Written: {i + 1}/{len(frames)} frames")

        out.release()

        print(f"✅ Video created: {output_path}")
        return True

    def process_video_autonomous(self, max_frames=150):
        """Komplett autonome Video-zu-Line Art Verarbeitung"""
        print("🚀 STARTING AUTONOMOUS LINE ART CONVERSION")
        print("=" * 60)

        start_time = time.time()

        # 1. Extract Frames
        frames, fps = self.extract_frames_from_video(max_frames)
        if not frames:
            return False

        print(f"\n🎨 Processing {len(frames)} frames to Line Art...")
        print("⏳ This may take several minutes...")
        print()

        # 2. Process Frames zu Line Art
        processed_frames = []
        total_frames = len(frames)

        for frame_num, frame in enumerate(frames):
            print(
                f"🖼️  Processing Frame {frame_num + 1}/{total_frames}...", end=" ")

            # Line Art Conversion
            line_art_frame = self.process_frame_to_line_art(frame)
            processed_frames.append(line_art_frame)

            # Progress Info
            elapsed = time.time() - start_time
            avg_time = elapsed / (frame_num + 1)
            eta = (total_frames - frame_num - 1) * avg_time
            progress = (frame_num + 1) / total_frames * 100

            print(f"✅ ({progress:.1f}% done, ETA: {eta:.1f}s)")

        # 3. Create Output Video
        print(f"\n📹 Creating final Line Art video...")
        success = self.create_video_from_frames(
            processed_frames, fps, self.output_path)

        total_time = time.time() - start_time

        if success:
            print(f"\n🎉 AUTONOMOUS LINE ART CONVERSION COMPLETE!")
            print("=" * 60)
            print(f"✅ Output Video: {self.output_path}")
            print(f"📊 Statistics:")
            print(f"   Total Frames: {len(processed_frames)}")
            print(f"   Processing Time: {total_time:.1f} seconds")
            print(
                f"   Average per Frame: {total_time/len(processed_frames):.2f} seconds")
            print(f"   Resolution: 512x512")
            print(f"   Frame Rate: {fps} FPS")
            print(f"   Style: Professional Line Art")
            print(f"   Method: Autonomous OpenCV+PIL")

            return True
        else:
            print(f"❌ Video creation failed")
            return False

    def create_comparison_gif(self):
        """Erstelle Vergleichs-GIF (Original vs Line Art)"""
        print("\n🎭 Creating comparison preview...")

        try:
            # Load original frames (first 10 for preview)
            frames, _ = self.extract_frames_from_video(max_frames=10)
            if not frames:
                return

            comparison_frames = []

            for frame in frames[:5]:  # First 5 frames
                # Original
                original = frame.resize((256, 256), Image.LANCZOS)

                # Line Art
                line_art = self.process_frame_to_line_art(
                    frame).resize((256, 256), Image.LANCZOS)

                # Create side-by-side comparison
                comparison = Image.new('RGB', (512, 256))
                comparison.paste(original, (0, 0))
                comparison.paste(line_art, (256, 0))

                comparison_frames.append(comparison)

            # Save as GIF
            gif_path = "output/COMPARISON_PREVIEW.gif"
            comparison_frames[0].save(
                gif_path,
                save_all=True,
                append_images=comparison_frames[1:],
                duration=800,
                loop=0
            )

            print(f"✅ Comparison preview: {gif_path}")

        except Exception as e:
            print(f"⚠️  Comparison creation failed: {e}")


def main():
    """Hauptfunktion für autonome Verarbeitung"""
    print("🎬 AUTONOMOUS VIDEO-TO-LINE ART PROCESSOR")
    print("=" * 60)
    print("🎯 Target: Schlossgifdance.mp4 → Professional Line Art")
    print("⚙️  Method: Direct OpenCV + PIL (No ComfyUI needed)")
    print("🚀 Mode: Completely Autonomous")
    print()

    processor = AutonomousVideoLineArtProcessor()

    # Check if input video exists
    if not Path(processor.input_video).exists():
        print(f"❌ Input video not found: {processor.input_video}")
        print("💡 Please ensure Schlossgifdance.mp4 is in the input/ directory")
        return

    print(f"✅ Input video found: {processor.input_video}")
    print()

    # Process mit 150 frames (reasonable processing time)
    success = processor.process_video_autonomous(max_frames=150)

    if success:
        print(f"\n🎉 SUCCESS! Autonomous Line Art conversion complete!")
        print(f"📁 Output: {processor.output_path}")
        print(f"🎨 Style: Professional Clean Line Art")
        print(f"📊 Quality: Production Ready")

        # Create comparison preview
        processor.create_comparison_gif()

        print(f"\n🎬 Ready to view your Line Art animation! ✨")
    else:
        print(f"\n❌ Autonomous processing failed.")
        print(f"🔧 Check input video and try again")


if __name__ == "__main__":
    main()
